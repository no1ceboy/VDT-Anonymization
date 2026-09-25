"""Evaluate a trained entity-linking checkpoint on a pair JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import MENTION_CLOSE, MENTION_OPEN
from .location_constraints import has_explicit_province_conflict
from .training import (
    EntityLinkingModel,
    JsonlPairDataset,
    PairCollator,
    binary_metrics,
    evaluate as score_model,
    save_json,
)
from .finetuning import ADAPTER_MODES, build_encoder, compute_dtype


def _load_pairs(path: Path) -> list[dict]:
    """Read every pair record in file order.

    The loader below uses ``shuffle=False`` over a ``JsonlPairDataset`` whose
    offsets are built in file order, so this re-read aligns positionally with
    the ``labels``/``scores`` that :func:`training.evaluate` returns.
    """
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def _grouped_metrics(labels: list[int], scores: list[float], threshold: float,
                      groups: list[str]) -> dict[str, dict]:
    by_group: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for label, score, group in zip(labels, scores, groups):
        group_labels, group_scores = by_group[group]
        group_labels.append(label)
        group_scores.append(score)
    return {
        name: binary_metrics(group_labels, group_scores, threshold)
        for name, (group_labels, group_scores) in sorted(by_group.items())
    }


def _link_error_summary(labels: list[int], scores: list[float], threshold: float,
                        vetoes: list[bool] | None = None) -> dict:
    """Report false merges directly, including for negative-only challenge sets."""
    vetoes = vetoes or [False] * len(labels)
    negative_count = sum(label == 0 for label in labels)
    false_links = sum(
        label == 0 and score >= threshold and not veto
        for label, score, veto in zip(labels, scores, vetoes)
    )
    return {
        "different_entity_pairs": negative_count,
        "false_links": false_links,
        "correctly_kept_separate": negative_count - false_links,
        "false_link_rate": false_links / negative_count if negative_count else None,
        "same_entity_pairs": sum(label == 1 for label in labels),
    }


def _mention_summary(mention: dict) -> dict:
    return {"surface": mention.get("surface"), "context": mention.get("context")}


def _example_entry(pair: dict, label: int, score: float, threshold: float,
                   location_veto: bool = False) -> dict:
    return {
        "pair_id": pair.get("pair_id"),
        "doc_id": pair.get("doc_id"),
        "difficulty": pair.get("difficulty"),
        "document_metadata": pair.get("document_metadata"),
        "mention_a": _mention_summary(pair.get("mention_a") or {}),
        "mention_b": _mention_summary(pair.get("mention_b") or {}),
        "target_linked": int(label),
        "predicted_score": round(float(score), 4),
        "model_predicted_linked": int(score >= threshold),
        "predicted_linked": int(score >= threshold and not location_veto),
        "decision_override": "conflicting_explicit_provinces" if location_veto else None,
    }


def _select_examples(pairs: list[dict], labels: list[int], scores: list[float],
                      threshold: float, max_examples: int,
                      location_vetoes: list[bool] | None = None) -> dict[str, list[dict]]:
    """Pick a small, easy-to-copy sample of predictions to inspect by hand.

    Meant for a workstation where pulling the full pairs/checkpoint back is
    impractical: this is a handful of short text snippets, not a dataset
    dump. False positives/negatives are sorted by how confidently the model
    got them wrong -- those are the most informative to read first.
    """
    if max_examples <= 0:
        return {}
    vetoes = location_vetoes or [False] * len(pairs)
    rows = list(zip(pairs, labels, scores, vetoes))
    false_positives = sorted(
        (row for row in rows if row[1] == 0 and row[2] >= threshold and not row[3]),
        key=lambda row: -row[2],
    )[:max_examples]
    false_negatives = sorted(
        (row for row in rows if row[1] == 1 and (row[2] < threshold or row[3])),
        key=lambda row: row[2],
    )[:max_examples]
    true_positives = [row for row in rows if row[1] == 1 and row[2] >= threshold and not row[3]][:max_examples]
    true_negatives = [row for row in rows if row[1] == 0 and (row[2] < threshold or row[3])][:max_examples]
    return {
        "false_positives_most_confident": [_example_entry(*row[:3], threshold, row[3]) for row in false_positives],
        "false_negatives_most_confident": [_example_entry(*row[:3], threshold, row[3]) for row in false_negatives],
        "true_positives_sample": [_example_entry(*row[:3], threshold, row[3]) for row in true_positives],
        "true_negatives_sample": [_example_entry(*row[:3], threshold, row[3]) for row in true_negatives],
    }


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

    mode = config.get("finetune_mode", "fft")
    amp_dtype = compute_dtype(args.amp_dtype)
    qlora_dtype = compute_dtype(args.qlora_compute_dtype) if args.qlora_compute_dtype else amp_dtype
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has no pad or eos token; provide a tokenizer with padding support")
        tokenizer.pad_token = tokenizer.eos_token
    if mode != "qlora":
        tokenizer.add_special_tokens({"additional_special_tokens": [MENTION_OPEN, MENTION_CLOSE]})
    adapter_path = None
    if mode in ADAPTER_MODES:
        adapter_path = args.checkpoint.parent / checkpoint.get("adapter_dir", "best_adapter")
        if not adapter_path.is_dir():
            raise ValueError(f"adapter checkpoint directory is missing: {adapter_path}")
    encoder, _ = build_encoder(
        str(model_source), mode, device,
        tokenizer_size=len(tokenizer) if mode != "qlora" else None,
        add_tokens=mode != "qlora",
        lora_target_modules=",".join(config.get("lora_target_modules") or []) or "auto",
        adapter_path=adapter_path,
        qlora_compute_dtype=qlora_dtype,
    )
    feature_names = tuple(config.get("feature_names") or [])
    dropout = float((config.get("arguments") or {}).get("dropout", 0.15))
    model = EntityLinkingModel(encoder, len(feature_names), dropout)
    if mode in ADAPTER_MODES:
        model.classifier.load_state_dict(checkpoint["classifier_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])
    if mode == "qlora":
        model.classifier.to(device)
    else:
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
    labels, scores, loss = score_model(model, loader, device, args.fp16, amp_dtype, args.max_eval_steps)
    threshold = float(checkpoint["threshold"])
    pairs = _load_pairs(args.pairs)
    scored = len(labels)
    if args.max_eval_steps is None and len(pairs) != scored:
        raise ValueError(
            f"pair count ({len(pairs)}) does not match scored example count ({scored}); "
            "the pairs file may have changed since the loader was built"
        )
    pairs = pairs[:scored]
    location_vetoes = [has_explicit_province_conflict(pair) for pair in pairs]
    constrained_scores = [
        min(float(score), -1.0) if veto else float(score)
        for score, veto in zip(scores, location_vetoes)
    ]
    difficulties = [str(pair.get("difficulty", "Unknown")) for pair in pairs]
    challenges = [str((pair.get("document_metadata") or {}).get("primary_challenge", "Unknown")) for pair in pairs]
    categories = [str((pair.get("document_metadata") or {}).get("category", "Unknown")) for pair in pairs]
    result = {
        "checkpoint": str(args.checkpoint),
        "pairs": str(args.pairs),
        "loss": loss,
        "decision_policy": {
            "location_veto": "force LOC pairs with different explicit provinces unlinked",
            "location_veto_count": sum(location_vetoes),
        },
        "metrics": {
            "overall": binary_metrics(labels, constrained_scores, threshold),
            "by_difficulty": _grouped_metrics(labels, constrained_scores, threshold, difficulties),
            "by_challenge": _grouped_metrics(labels, constrained_scores, threshold, challenges),
            "by_category": _grouped_metrics(labels, constrained_scores, threshold, categories),
        },
        "model_only_metrics": {
            "overall": binary_metrics(labels, scores, threshold),
            "by_difficulty": _grouped_metrics(labels, scores, threshold, difficulties),
            "by_challenge": _grouped_metrics(labels, scores, threshold, challenges),
            "by_category": _grouped_metrics(labels, scores, threshold, categories),
        },
        "examples": _select_examples(
            pairs, labels, scores, threshold, args.max_examples, location_vetoes
        ),
        "weak_supervision_warning": "Metrics measure agreement with rule-generated pair labels.",
    }
    if args.collision_pairs:
        collision_dataset = JsonlPairDataset(args.collision_pairs)
        collision_loader = DataLoader(
            collision_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        collision_labels, collision_scores, collision_loss = score_model(
            model, collision_loader, device, args.fp16, amp_dtype, args.max_eval_steps
        )
        collision_pairs = _load_pairs(args.collision_pairs)
        if args.max_eval_steps is None and len(collision_pairs) != len(collision_labels):
            raise ValueError(
                "collision pair count does not match scored example count; "
                "the pair file may have changed since the loader was built"
            )
        collision_pairs = collision_pairs[:len(collision_labels)]
        collision_vetoes = [has_explicit_province_conflict(pair) for pair in collision_pairs]
        collision_constrained_scores = [
            min(float(score), -1.0) if veto else float(score)
            for score, veto in zip(collision_scores, collision_vetoes)
        ]
        collision_difficulties = [str(pair.get("difficulty", "Unknown")) for pair in collision_pairs]
        collision_challenges = [
            str((pair.get("document_metadata") or {}).get("primary_challenge", "Unknown"))
            for pair in collision_pairs
        ]
        collision_categories = [
            str((pair.get("document_metadata") or {}).get("category", "Unknown"))
            for pair in collision_pairs
        ]
        collision_warning = None
        if not any(collision_labels):
            collision_warning = (
                "This collision file has no same-entity pairs; use false_links and "
                "false_link_rate to measure bad merges. Positive-class F1 is not informative here."
            )
        result["collision_evaluation"] = {
            "pairs": str(args.collision_pairs),
            "loss": collision_loss,
            "decision_policy": {
                "location_veto_count": sum(collision_vetoes),
                "threshold": threshold,
            },
            "metrics": {
                "overall": binary_metrics(collision_labels, collision_constrained_scores, threshold),
                "by_difficulty": _grouped_metrics(
                    collision_labels, collision_constrained_scores, threshold, collision_difficulties
                ),
                "by_challenge": _grouped_metrics(
                    collision_labels, collision_constrained_scores, threshold, collision_challenges
                ),
                "by_category": _grouped_metrics(
                    collision_labels, collision_constrained_scores, threshold, collision_categories
                ),
            },
            "model_only_metrics": binary_metrics(collision_labels, collision_scores, threshold),
            "link_errors_after_location_rule": _link_error_summary(
                collision_labels, collision_scores, threshold, collision_vetoes
            ),
            "link_errors_model_only": _link_error_summary(collision_labels, collision_scores, threshold),
            "warning": collision_warning,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--collision-pairs", type=Path,
                        help="optional separate hard-collision pairs file; scored at the validation-selected threshold")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", help="override the encoder model ID/path stored in the checkpoint")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--qlora-compute-dtype", choices=["fp16", "bf16"])
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--max-examples", type=int, default=15,
                        help="qualitative examples per bucket to include in the output (0 disables)")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.pairs = args.pairs.resolve()
    if args.collision_pairs:
        args.collision_pairs = args.collision_pairs.resolve()
    args.output = args.output.resolve()
    if (not args.checkpoint.is_file() or not args.pairs.is_file()
            or (args.collision_pairs and not args.collision_pairs.is_file())):
        parser.error("checkpoint and all requested pair files must exist")
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_length is not None and args.max_length < 16):
        parser.error("batch size must be positive, workers non-negative, and max length >=16")
    if args.max_eval_steps is not None and args.max_eval_steps < 1:
        parser.error("max eval steps must be positive")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
