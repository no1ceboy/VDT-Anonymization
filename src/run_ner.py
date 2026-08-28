"""Run a Vietnamese NER checkpoint over the legal JSONL dataset.

The default checkpoint is NlpHUST/ner-vietnamese-electra-base.  The script
uses Hugging Face Transformers, but supports a fully local checkpoint for the
air-gapped machine.  It performs sliding-window inference and preserves
character offsets in the original document text.

Examples:

    # Download/cache or resolve a Hugging Face model reference
    python src/run_ner.py \
        --model-source huggingface \
        --model-name NlpHUST/ner-vietnamese-electra-base \
        --input-file datasets/legal_test.jsonl \
        --output-file outputs/nlphust_legal_test.jsonl \
        --limit 100

    # Later, use a manually uploaded local checkpoint
    python src/run_ner.py \
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

    encoded = tokenizer(
        text,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=True,
        return_tensors="pt",
    )
    offsets = encoded["offset_mapping"]
    special_tokens = encoded.pop("special_tokens_mask")
    encoded.pop("offset_mapping")
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    all_entities = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, model_inputs["input_ids"].shape[0], batch_size):
            batch = {
                key: value[batch_start:batch_start + batch_size]
                for key, value in model_inputs.items()
            }
            output = model(**batch)
            probabilities = torch.softmax(output.logits, dim=-1)
            scores, predictions = probabilities.max(dim=-1)

            for local_index in range(predictions.shape[0]):
                chunk_offsets = offsets[batch_start + local_index].tolist()
                chunk_special = special_tokens[batch_start + local_index].tolist()
                chunk_predictions = predictions[local_index].tolist()
                chunk_scores = scores[local_index].tolist()
                current = None

                for token_index, (start_end, token_id, token_score, is_special) in enumerate(
                    zip(chunk_offsets, chunk_predictions, chunk_scores, chunk_special)
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
    parser.add_argument("--text-field", default="markdown", help="Input row field containing document text")
    parser.add_argument("--id-field", default="case_id", help="Input row field used as document ID")
    parser.add_argument("--entity-types", default="PER,LOC",
                        help="Comma-separated entity types, or 'all' (default: PER,LOC)")
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

    try:
        entity_types = parse_entity_types(args.entity_types)
        device = choose_device(args.device)
        tokenizer, model, model_reference = load_model(args, device)
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to load NER model: {type(exc).__name__}: {exc}")

    label_map = model_label_map(model)
    print(f"[NER] Device: {device}; labels: {label_map}")
    print(f"[NER] Entity filter: {sorted(entity_types) if entity_types else 'all'}")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)
    counts = Counter()
    output_documents = 0
    output_entities = 0
    errors = 0

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
                    entities = infer_document(
                        text=text,
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
    print(f"[NER] Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
