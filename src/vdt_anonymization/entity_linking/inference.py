"""Document-level mention scoring and conservative entity clustering."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .clustering import complete_link_clusters
from .dataset import MENTION_CLOSE, MENTION_OPEN
from .finetuning import ADAPTER_MODES, build_encoder, compute_dtype
from .location_constraints import location_rule_decision
from .raw_dataset import RAW_PAIR_FEATURE_NAMES, raw_pair_features
from .training import EntityLinkingModel, save_json


def _prepare_mentions(text: str, mentions: list[dict], context_chars: int) -> list[dict]:
    prepared = []
    for index, raw in enumerate(mentions):
        try:
            start, end = int(raw["start"]), int(raw["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"mention {index} must have integer start/end offsets") from error
        if not 0 <= start < end <= len(text):
            raise ValueError(f"mention {index} span [{start}, {end}) is outside the document")
        surface = text[start:end]
        supplied_surface = raw.get("surface")
        if supplied_surface is not None and str(supplied_surface) != surface:
            raise ValueError(f"mention {index} surface does not match text[start:end]")
        label = str(raw.get("label") or "").strip()
        if not label:
            raise ValueError(f"mention {index} has no entity label/type")
        left, right = max(0, start - context_chars), min(len(text), end + context_chars)
        prepared.append({
            **raw,
            "label": label,
            "start": start,
            "end": end,
            "surface": surface,
            "context": text[left:start] + MENTION_OPEN + surface + MENTION_CLOSE + text[end:right],
        })
    return sorted(prepared, key=lambda item: (item["start"], item["end"], item["label"]))


class DocumentEntityLinker:
    """Load a trained pair scorer and apply it to all mentions in each document."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str | None = None,
        tokenizer_path: Path | None = None,
        device: str = "auto",
        fp16: bool = True,
        amp_dtype: str = "bf16",
        qlora_compute_dtype: str | None = None,
        mention_batch_size: int = 32,
        pair_batch_size: int = 4096,
        max_length: int | None = None,
        context_chars: int = 160,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        requested_device = device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.fp16 = fp16
        self.amp_dtype = compute_dtype(amp_dtype)
        self.qlora_dtype = compute_dtype(qlora_compute_dtype) if qlora_compute_dtype else self.amp_dtype
        self.mention_batch_size = mention_batch_size
        self.pair_batch_size = pair_batch_size
        self.context_chars = context_chars

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if "threshold" not in checkpoint:
            raise ValueError("checkpoint has no validation threshold; expected best_model.pt")
        self.checkpoint = checkpoint
        self.config = checkpoint.get("config") or {}
        self.threshold = float(checkpoint["threshold"])
        self.model_name = model_name or self.config.get("model_name")
        if not self.model_name:
            raise ValueError("model source missing; pass --model-name")
        self.mode = self.config.get("finetune_mode", "fft")
        self.max_length = max_length or int(self.config.get("max_length", 256))

        source = tokenizer_path or self.checkpoint_path.parent / "tokenizer"
        if not Path(source).exists():
            source = self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(str(source))
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("tokenizer has no pad or eos token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        add_tokens = self.mode != "qlora"
        if add_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [MENTION_OPEN, MENTION_CLOSE]})

        adapter_path = None
        if self.mode in ADAPTER_MODES:
            adapter_path = self.checkpoint_path.parent / checkpoint.get("adapter_dir", "best_adapter")
            if not adapter_path.is_dir():
                raise ValueError(f"adapter checkpoint directory missing: {adapter_path}")
        encoder, _ = build_encoder(
            str(self.model_name), self.mode, self.device,
            tokenizer_size=len(self.tokenizer) if add_tokens else None,
            add_tokens=add_tokens,
            lora_target_modules=",".join(self.config.get("lora_target_modules") or []) or "auto",
            adapter_path=adapter_path,
            qlora_compute_dtype=self.qlora_dtype,
        )
        feature_names = tuple(self.config.get("feature_names") or RAW_PAIR_FEATURE_NAMES)
        dropout = float((self.config.get("arguments") or {}).get("dropout", 0.15))
        self.feature_names = feature_names
        self.model = EntityLinkingModel(encoder, len(feature_names), dropout)
        if self.mode in ADAPTER_MODES:
            self.model.classifier.load_state_dict(checkpoint["classifier_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])
        if self.mode == "qlora":
            self.model.classifier.to(self.device)
        else:
            self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_document_pairs(self, text: str, mentions: list[dict]) -> tuple[
        list[dict], dict[tuple[int, int], float], set[tuple[int, int]], set[tuple[int, int]]
    ]:
        prepared = _prepare_mentions(text, mentions, self.context_chars)
        if len(prepared) < 2:
            return prepared, {}, set(), set()

        # Encode every mention context once, then score all same-type pairs.
        embeddings = []
        for start in range(0, len(prepared), self.mention_batch_size):
            batch_mentions = prepared[start:start + self.mention_batch_size]
            encoded = self.tokenizer(
                [mention["context"] for mention in batch_mentions],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.fp16 and self.device.type == "cuda",
            ):
                embeddings.append(self.model.encode(encoded))
        embeddings = torch.cat(embeddings, dim=0)

        pairs = [
            (left, right)
            for left, right in combinations(range(len(prepared)), 2)
            if prepared[left]["label"].casefold() == prepared[right]["label"].casefold()
        ]
        if not pairs:
            return prepared, {}, set(), set()

        scores: dict[tuple[int, int], float] = {}
        cannot_link: set[tuple[int, int]] = set()
        must_link: set[tuple[int, int]] = set()
        for offset in range(0, len(pairs), self.pair_batch_size):
            pair_batch = pairs[offset:offset + self.pair_batch_size]
            left_indices = torch.tensor([pair[0] for pair in pair_batch], device=self.device)
            right_indices = torch.tensor([pair[1] for pair in pair_batch], device=self.device)
            features = torch.tensor(
                [[float(raw_pair_features(text, prepared[left], prepared[right]).get(name, 0.0))
                  for name in self.feature_names]
                 for left, right in pair_batch],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.fp16 and self.device.type == "cuda",
            ):
                logits = self.model.classify_embeddings(
                    embeddings[left_indices], embeddings[right_indices], features
                )
            probabilities = torch.sigmoid(logits).float().cpu().tolist()
            for (left, right), probability in zip(pair_batch, probabilities):
                scores[(left, right)] = float(probability)
                pair = {"mention_a": prepared[left], "mention_b": prepared[right]}
                rule_decision = location_rule_decision(pair)
                if rule_decision == "block":
                    cannot_link.add((left, right))
                elif rule_decision == "link":
                    must_link.add((left, right))
        return prepared, scores, cannot_link, must_link

    @torch.no_grad()
    def predict_document(self, doc_id: str, text: str, mentions: list[dict],
                         threshold: float | None = None) -> dict:
        prepared, scores, cannot_link, must_link = self.score_document_pairs(text, mentions)
        selected_threshold = self.threshold if threshold is None else float(threshold)
        clusters = complete_link_clusters(
            prepared, scores, selected_threshold, cannot_link, must_link
        )
        linked_mentions = []
        output_clusters = []
        for cluster_number, indices in enumerate(clusters, start=1):
            cluster_id = f"{doc_id}:E{cluster_number:04d}"
            members = []
            for index in indices:
                mention = {key: value for key, value in prepared[index].items()
                           if key not in {"context", "gold_entity_id", "entity_id"}}
                mention["cluster_id"] = cluster_id
                linked_mentions.append(mention)
                members.append(mention)
            output_clusters.append({"cluster_id": cluster_id, "label": members[0]["label"], "mentions": members})
        return {
            "doc_id": str(doc_id),
            "threshold": selected_threshold,
            "clustering": "complete_link_with_location_rules",
            "candidate_pairs_scored": len(scores),
            "location_rule_links": len(must_link),
            "location_rule_blocks": len(cannot_link),
            "clusters": output_clusters,
            "mentions": sorted(linked_mentions, key=lambda item: (item["start"], item["end"])),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL: each row has doc_id, text, and mentions [{start,end,label,...}]")
    parser.add_argument("--output", type=Path, required=True, help="JSONL cluster predictions")
    parser.add_argument("--model-name")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", default="auto")
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--threshold", type=float,
                                 help="override the checkpoint threshold")
    threshold_group.add_argument("--threshold-file", type=Path,
                                 help="cluster_validation.json created by cluster_evaluate --select-threshold")
    parser.add_argument("--mention-batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable mixed-precision autocast; --amp-dtype chooses its dtype")
    parser.add_argument("--qlora-compute-dtype", choices=["fp16", "bf16"])
    args = parser.parse_args()
    if args.mention_batch_size < 1 or args.pair_batch_size < 1 or args.context_chars < 1:
        parser.error("batch sizes and context chars must be positive")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    for name in ("checkpoint", "input"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, path)
    args.output = args.output.resolve()
    if args.tokenizer:
        args.tokenizer = args.tokenizer.resolve()
    if args.threshold_file:
        args.threshold_file = args.threshold_file.resolve()
        if not args.threshold_file.is_file():
            parser.error(f"threshold file does not exist: {args.threshold_file}")
    return args


def main() -> None:
    args = parse_args()
    linker = DocumentEntityLinker(
        args.checkpoint, args.model_name, args.tokenizer, args.device,
        not args.no_fp16, args.amp_dtype, args.qlora_compute_dtype,
        args.mention_batch_size, args.pair_batch_size, args.max_length, args.context_chars,
    )
    threshold = args.threshold
    if args.threshold_file:
        threshold = float(json.loads(args.threshold_file.read_text(encoding="utf-8"))["threshold"])
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold from file must be between 0 and 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8-sig") as source, args.output.open("w", encoding="utf-8") as sink:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("document_text")
            if not isinstance(text, str):
                raise ValueError(f"input line {line_number} has no text/document_text")
            result = linker.predict_document(
                str(row.get("doc_id") or row.get("id") or line_number), text,
                row.get("mentions") or [], threshold,
            )
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
