"""Evaluate a trained entity-linking checkpoint on a pair JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .dataset import MENTION_CLOSE, MENTION_OPEN
from .training import (
    EntityLinkingModel,
    JsonlPairDataset,
    PairCollator,
    binary_metrics,
    evaluate as score_model,
    save_json,
)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "threshold" not in checkpoint:
        raise ValueError("checkpoint has no validation-selected threshold; use best_model.pt")
    config = checkpoint.get("config") or {}
    model_source = args.model_name or config.get("model_name")
    if not model_source:
        raise ValueError("model source is missing; pass --model-name")
    tokenizer_source = args.tokenizer or args.checkpoint.parent / "tokenizer"
    if isinstance(tokenizer_source, Path) and not tokenizer_source.exists():
        tokenizer_source = model_source

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    tokenizer.add_special_tokens({"additional_special_tokens": [MENTION_OPEN, MENTION_CLOSE]})
    encoder = AutoModel.from_pretrained(str(model_source))
    encoder.resize_token_embeddings(len(tokenizer))
    feature_names = tuple(config.get("feature_names") or [])
    dropout = float((config.get("arguments") or {}).get("dropout", 0.15))
    model = EntityLinkingModel(encoder, len(feature_names), dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = JsonlPairDataset(args.pairs)
    collator = PairCollator(tokenizer, args.max_length or int(config.get("max_length", 256)), feature_names)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    labels, scores, loss = score_model(model, loader, device, args.fp16)
    result = {
        "checkpoint": str(args.checkpoint),
        "pairs": str(args.pairs),
        "loss": loss,
        "metrics": binary_metrics(labels, scores, float(checkpoint["threshold"])),
        "weak_supervision_warning": "Metrics measure agreement with rule-generated pair labels.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", help="override the encoder model ID/path stored in the checkpoint")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.pairs = args.pairs.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_file() or not args.pairs.is_file():
        parser.error("checkpoint and pair files must exist")
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_length is not None and args.max_length < 16):
        parser.error("batch size must be positive, workers non-negative, and max length >=16")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
