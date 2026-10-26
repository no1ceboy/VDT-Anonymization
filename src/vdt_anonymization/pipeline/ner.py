"""Run a Vietnamese NER checkpoint over the legal JSONL dataset.

The default checkpoint is NlpHUST/ner-vietnamese-electra-base.  The script
uses Hugging Face Transformers, but supports a fully local checkpoint for the
air-gapped machine.  It performs sliding-window inference and preserves
character offsets in the original document text.

Examples:

    # Download/cache or resolve a Hugging Face model reference
    vdt run-ner \
        --model-source huggingface \
        --model-name NlpHUST/ner-vietnamese-electra-base \
        --input-file datasets/legal_test.jsonl \
        --output-file outputs/nlphust_legal_test.jsonl \
        --limit 100

    # Later, use a manually uploaded local checkpoint
    vdt run-ner \
        --model-source local \
        --model-path /models/ner-vietnamese-electra-base \
        --input-file datasets/legal_test.jsonl \
        --output-file outputs/nlphust_legal_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean

from .ocr_ngram import CharacterTrigramModel, MODEL_FORMAT, normalize_before_ner


DEFAULT_MODEL = "NlpHUST/ner-vietnamese-electra-base"
DEFAULT_INPUT = "datasets/legal_test.jsonl"
DEFAULT_OUTPUT = "outputs/nlphust_legal_test.jsonl"

# Checkpoint label names are not universal. NlpHUST uses PERSON/LOCATION,
# while many NER interfaces and our CLI use PER/LOC. Canonicalize both forms
# before applying --entity-types so valid predictions are not filtered out.
ENTITY_TYPE_ALIASES = {
    "PER": "PER",
    "PERSON": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "MISC": "MISC",
    "MISCELLANEOUS": "MISC",
}


def load_jsonl_rows(path, offset=0, limit=0):
    """Yield rows without loading the 800 MB legal file into memory."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle):
            if row_number < offset:
                continue
            if limit > 0 and row_number >= offset + limit:
                break
            line = line.strip()
            if not line:
                continue
            yield row_number, json.loads(line)


def choose_device(requested):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_entity_types(value):
    if value.strip().lower() in {"all", "*"}:
        return None
    values = {
        ENTITY_TYPE_ALIASES.get(part.strip().upper(), part.strip().upper())
        for part in value.split(",")
        if part.strip()
    }
    if not values:
        raise ValueError("--entity-types must contain at least one type or 'all'")
    return values


def split_label(raw_label):
    """Convert BIO/BIOES and plain labels into (prefix, entity type)."""
    label = str(raw_label or "O").strip()
    if label.upper() in {"O", "OUTSIDE", "LABEL_0"}:
        return "O", "O"
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
        prefix = prefix.upper()
        if prefix in {"B", "I", "E", "S", "U"}:
            normalized_type = ENTITY_TYPE_ALIASES.get(entity_type.upper(), entity_type.upper())
            return prefix, normalized_type
    return "I", ENTITY_TYPE_ALIASES.get(label.upper(), label.upper())


def model_label_map(model):
    id2label = getattr(model.config, "id2label", {}) or {}
    return {int(key): value for key, value in id2label.items()}


def append_entity(entities, current):
    if current is None:
        return
    if current["end"] <= current["start"]:
        return
    entities.append({
        "text": current["text"],
        "start": current["start"],
        "end": current["end"],
        "label": current["label"],
        "score": round(mean(current["scores"]), 6),
    })


def infer_document(text, tokenizer, model, device, label_map, entity_types,
                   max_length, stride, batch_size, min_confidence):
    """Infer entities and return character offsets relative to `text`."""
    import torch

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if stride < 0:
        raise ValueError("stride cannot be negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not text:
        return []

    # Tokenize once without truncation, then create every model window ourselves.
    # Some fast-tokenizer overflow encodings can omit later windows for a single
    # long string. Manual slicing makes full-document coverage explicit.
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = list(encoded["input_ids"])
    token_offsets = encoded["offset_mapping"]
    empty_encoding = tokenizer("", add_special_tokens=True, truncation=False)
    special_count = len(empty_encoding["input_ids"])
    content_length = max_length - special_count
    if content_length <= 0:
        raise ValueError("max_length leaves no room for text tokens after special tokens")
    if stride >= content_length:
        raise ValueError("stride must be smaller than the content window length")
    step = content_length - stride

    windows = []
    allowed_inputs = set(getattr(tokenizer, "model_input_names", ["input_ids", "attention_mask"]))
    for start in range(0, len(token_ids), step):
        end = min(start + content_length, len(token_ids))
        window_offsets = token_offsets[start:end]
        valid_offsets = [
            (int(pair[0]), int(pair[1])) for pair in window_offsets
            if int(pair[1]) > int(pair[0])
        ]
        if not valid_offsets:
            if end == len(token_ids):
                break
            continue
        char_start = min(pair[0] for pair in valid_offsets)
        char_end = max(pair[1] for pair in valid_offsets)
        window_text = text[char_start:char_end]
        prepared = tokenizer(
            window_text, add_special_tokens=True, truncation=False,
            return_offsets_mapping=True, return_special_tokens_mask=True,
            return_attention_mask=True, return_token_type_ids=True,
        )
        input_ids = list(prepared["input_ids"])
        special_mask = list(prepared["special_tokens_mask"])
        offsets = []
        for pair, is_special in zip(prepared["offset_mapping"], special_mask):
            if is_special:
                offsets.append((0, 0))
            else:
                offsets.append((char_start + int(pair[0]), char_start + int(pair[1])))
        if len(offsets) != len(input_ids) or len(input_ids) > max_length:
            raise RuntimeError("A manually built NER window exceeded max_length")
        features = {
            key: value for key, value in prepared.items()
            if key in allowed_inputs
        }
        windows.append({
            "features": features,
            "offsets": offsets,
            "special_tokens_mask": special_mask,
        })
        if end == len(token_ids):
            break

    all_entities = []
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start:batch_start + batch_size]
            features = [window["features"] for window in batch_windows]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            padded_length = int(batch["input_ids"].shape[1])
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            output = model(**model_inputs)
            probabilities = torch.softmax(output.logits, dim=-1)
            scores, predictions = probabilities.max(dim=-1)

            for local_index, window in enumerate(batch_windows):
                chunk_offsets = list(window["offsets"])
                chunk_special = list(window["special_tokens_mask"])
                padding_count = padded_length - len(chunk_offsets)
                if padding_count > 0:
                    chunk_offsets.extend([(0, 0)] * padding_count)
                    chunk_special.extend([1] * padding_count)
                chunk_predictions = predictions[local_index].tolist()
                chunk_scores = scores[local_index].tolist()
                current = None

                for start_end, token_id, token_score, is_special in zip(
                    chunk_offsets, chunk_predictions, chunk_scores, chunk_special
                ):
                    start, end = int(start_end[0]), int(start_end[1])
                    if is_special or end <= start or start < 0 or end > len(text):
                        continue

                    raw_label = label_map.get(int(token_id), "O")
                    prefix, entity_type = split_label(raw_label)
                    confidence = float(token_score)
                    accepted = entity_type != "O" and (
                        entity_types is None or entity_type in entity_types
                    ) and confidence >= min_confidence

                    if not accepted:
                        append_entity(all_entities, current)
                        current = None
                        continue

                    starts_new = (
                        current is None
                        or prefix in {"B", "S", "U", "E"}
                        or current["label"] != entity_type
                        or start < current["start"]
                        or start > current["end"] + 2
                    )
                    if starts_new:
                        append_entity(all_entities, current)
                        current = {
                            "text": text[start:end],
                            "start": start,
                            "end": end,
                            "label": entity_type,
                            "scores": [confidence],
                        }
                    else:
                        current["end"] = max(current["end"], end)
                        current["text"] = text[current["start"]:current["end"]]
                        current["scores"].append(confidence)
                        if prefix in {"E", "S", "U"}:
                            append_entity(all_entities, current)
                            current = None

                append_entity(all_entities, current)

    # Overlapping windows produce duplicate spans. Keep the highest-scoring
    # copy for each exact (start, end, label) span.
    deduplicated = {}
    for entity in all_entities:
        key = (entity["start"], entity["end"], entity["label"])
        old = deduplicated.get(key)
        if old is None or entity["score"] > old["score"]:
            deduplicated[key] = entity

    return sorted(deduplicated.values(), key=lambda item: (item["start"], item["end"], item["label"]))


def load_model(args, device):
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    if args.model_source == "local":
        if not args.model_path:
            raise ValueError("--model-path is required when --model-source local")
        model_reference = args.model_path
        local_files_only = True
    else:
        model_reference = args.model_name
        local_files_only = args.local_files_only

    print(f"[NER] Loading tokenizer/model: {model_reference}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("This script requires a fast tokenizer for character offsets")

    model_kwargs = {"local_files_only": local_files_only}
    if device.type == "cuda" and args.dtype != "fp32":
        selected_dtype = "bf16" if args.dtype == "auto" else args.dtype
        model_kwargs["torch_dtype"] = torch.bfloat16 if selected_dtype == "bf16" else torch.float16
    model = AutoModelForTokenClassification.from_pretrained(model_reference, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model, model_reference


def get_document_id(row, id_field, row_number):
    value = row.get(id_field)
    if value is not None and str(value).strip():
        return str(value)
    for fallback in ("case_id", "doc_name", "doc_code", "id"):
        value = row.get(fallback)
        if value is not None and str(value).strip():
            return str(value)
    return f"row_{row_number}"


def main():
    parser = argparse.ArgumentParser(description="Run Vietnamese NER over legal JSONL documents")
    parser.add_argument("--model-source", choices=["huggingface", "local"], default="huggingface",
                        help="Resolve a Hugging Face model reference or a local uploaded checkpoint")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help=f"Hugging Face model ID/path (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-path", default=None,
                        help="Local checkpoint directory; required with --model-source local")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Do not attempt downloads when using --model-source huggingface")
    parser.add_argument("--input-file", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT, help="Output prediction JSONL file")
    parser.add_argument("--ocr-ngram-model", default=None,
                        help="Optional corpus-trained model for conservative pre-NER spacing normalization")
    parser.add_argument("--ocr-min-ngram-gain", type=float, default=0.18,
                        help="Minimum mean character-trigram score gain required to join fragments")
    parser.add_argument("--ocr-min-candidate-frequency", type=int, default=3,
                        help="Minimum corpus count required for a joined token")
    parser.add_argument("--text-field", default="markdown", help="Input row field containing document text")
    parser.add_argument("--id-field", default="case_id", help="Input row field used as document ID")
    parser.add_argument("--entity-types", default="PER,LOC,ORG",
                        help="Comma-separated entity types, or 'all' (default: PER,LOC,ORG; LOC is a rule-based fallback)")
    parser.add_argument("--limit", type=int, default=0, help="Number of rows to process; 0 means all")
    parser.add_argument("--offset", type=int, default=0, help="Number of JSONL rows to skip")
    parser.add_argument("--max-length", type=int, default=512, help="Tokens per sliding window")
    parser.add_argument("--stride", type=int, default=128, help="Overlap between windows")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of windows per model batch")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Discard entity tokens below this confidence")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="Inference dtype on CUDA; auto uses bf16")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        parser.error(f"Input file does not exist: {args.input_file}")
    if args.limit < 0 or args.offset < 0:
        parser.error("--limit and --offset cannot be negative")
    if args.max_length < 8:
        parser.error("--max-length must be at least 8")
    if args.stride < 0 or args.stride >= args.max_length - 2:
        parser.error("--stride must be non-negative and smaller than max-length - 2")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.min_confidence < 0 or args.min_confidence > 1:
        parser.error("--min-confidence must be between 0 and 1")
    if args.ocr_min_ngram_gain < 0:
        parser.error("--ocr-min-ngram-gain cannot be negative")
    if args.ocr_min_candidate_frequency < 1:
        parser.error("--ocr-min-candidate-frequency must be positive")
    if args.ocr_ngram_model and not os.path.isfile(args.ocr_ngram_model):
        parser.error(f"OCR n-gram model does not exist: {args.ocr_ngram_model}")

    try:
        entity_types = parse_entity_types(args.entity_types)
        device = choose_device(args.device)
        tokenizer, model, model_reference = load_model(args, device)
        ocr_model = CharacterTrigramModel.load(args.ocr_ngram_model) if args.ocr_ngram_model else None
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to load inference model(s): {type(exc).__name__}: {exc}")

    label_map = model_label_map(model)
    print(f"[NER] Device: {device}; labels: {label_map}")
    print(f"[NER] Entity filter: {sorted(entity_types) if entity_types else 'all'}")
    ocr_min_frequency = args.ocr_min_candidate_frequency
    if ocr_model:
        ocr_min_frequency = max(
            ocr_min_frequency,
            int(ocr_model.metadata.get("min_word_count", 1)),
        )
        print(f"[NER] OCR spacing model: {args.ocr_ngram_model}; format={MODEL_FORMAT}; "
              f"min_gain={args.ocr_min_ngram_gain:g}; "
              f"min_candidate_frequency={ocr_min_frequency}")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)
    counts = Counter()
    output_documents = 0
    output_entities = 0
    errors = 0
    ocr_whitespace_changes = 0
    ocr_join_count = 0

    with open(args.output_file, "w", encoding="utf-8") as output_handle:
        for processed_index, (row_number, row) in enumerate(
            load_jsonl_rows(args.input_file, args.offset, args.limit), 1
        ):
            document_id = get_document_id(row, args.id_field, row_number)
            raw_text = row.get(args.text_field)
            record = {
                "row_number": row_number,
                "doc_id": document_id,
                "model": model_reference,
                "text_field": args.text_field,
            }
            if raw_text is None:
                record.update({"char_len": 0, "entities": [], "error": f"missing field: {args.text_field}"})
                errors += 1
                print(f"[WARN] {document_id}: missing {args.text_field}")
            else:
                text = str(raw_text)
                record["char_len"] = len(text)
                try:
                    inference_text = text
                    normalization = None
                    if ocr_model:
                        normalization = normalize_before_ner(
                            text,
                            ocr_model,
                            min_gain=args.ocr_min_ngram_gain,
                            min_candidate_frequency=ocr_min_frequency,
                        )
                        inference_text = normalization.text
                    entities = infer_document(
                        text=inference_text,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        label_map=label_map,
                        entity_types=entity_types,
                        max_length=args.max_length,
                        stride=args.stride,
                        batch_size=args.batch_size,
                        min_confidence=args.min_confidence,
                    )
                    if normalization:
                        mapped_entities = []
                        for entity in entities:
                            source_span = normalization.source_span(entity["start"], entity["end"])
                            if source_span is None:
                                continue
                            entity["start"], entity["end"] = source_span
                            entity["text"] = text[source_span[0]:source_span[1]]
                            mapped_entities.append(entity)
                        entities = mapped_entities
                        record["ocr_normalization"] = {
                            "model_format": MODEL_FORMAT,
                            "model_file": os.path.basename(args.ocr_ngram_model),
                            "min_ngram_gain": args.ocr_min_ngram_gain,
                            "min_candidate_frequency": ocr_min_frequency,
                            "whitespace_changes": normalization.whitespace_changes,
                            "joined_fragments": len(normalization.joins or []),
                            "join_spans": normalization.joins or [],
                        }
                        ocr_whitespace_changes += normalization.whitespace_changes
                        ocr_join_count += len(normalization.joins or [])
                    record["entities"] = entities
                    output_entities += len(entities)
                    counts.update(entity["label"] for entity in entities)
                except Exception as exc:
                    record.update({"entities": [], "error": f"{type(exc).__name__}: {exc}"})
                    errors += 1
                    print(f"[WARN] {document_id}: {type(exc).__name__}: {exc}")

            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            output_documents += 1
            if processed_index == 1 or processed_index % 10 == 0:
                print(f"[NER] Processed {processed_index} documents; entities={output_entities}")

    print("\n[NER] Completed")
    print(f"[NER] Documents: {output_documents}; errors: {errors}")
    print(f"[NER] Entities: {output_entities}; by type: {dict(counts)}")
    if ocr_model:
        print(f"[NER] OCR spacing changes: {ocr_whitespace_changes}; joined fragments: {ocr_join_count}")
    print(f"[NER] Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
