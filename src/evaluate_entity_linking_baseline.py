"""Evaluate transparent rule baselines against weak entity-linking labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .entity_linking_data import marker_family, normalize_text
except ImportError:
    from entity_linking_data import marker_family, normalize_text


def predict(pair: dict, method: str) -> int:
    a, b = pair["mention_a"], pair["mention_b"]
    same_label = a.get("label") == b.get("label")
    if method == "exact_marker":
        marker_a, marker_b = normalize_text(a.get("marker")), normalize_text(b.get("marker"))
        return int(same_label and bool(marker_a) and marker_a == marker_b)
    if method == "marker_family":
        family_a, family_b = marker_family(a.get("marker")), marker_family(b.get("marker"))
        return int(same_label and bool(family_a) and family_a == family_b)
    if method == "exact_surface":
        return int(same_label and normalize_text(a.get("surface")) == normalize_text(b.get("surface")))
    raise ValueError(f"unknown baseline: {method}")


def metrics(confusion: Counter) -> dict:
    tp, fp, tn, fn = (confusion[key] for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "examples": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": {key: confusion[key] for key in ("tp", "fp", "tn", "fn")},
    }


def evaluate(path: Path, methods: list[str]) -> dict:
    totals = {method: Counter() for method in methods}
    by_difficulty = {method: defaultdict(Counter) for method in methods}
    by_challenge = {method: defaultdict(Counter) for method in methods}
    by_category = {method: defaultdict(Counter) for method in methods}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            target = int(pair["target_linked"])
            for method in methods:
                prediction = predict(pair, method)
                key = "tp" if target and prediction else "fn" if target else "fp" if prediction else "tn"
                totals[method][key] += 1
                by_difficulty[method][pair.get("difficulty", "Unknown")][key] += 1
                metadata = pair.get("document_metadata") or {}
                by_challenge[method][metadata.get("primary_challenge", "Unknown")][key] += 1
                by_category[method][metadata.get("category", "Unknown")][key] += 1
    return {
        "evaluation_kind": "agreement with weak rule-generated labels; not independent human-ground-truth accuracy",
        "pair_file": str(path),
        "baselines": {
            method: {
                "overall": metrics(totals[method]),
                "by_difficulty": {
                    name: metrics(values) for name, values in sorted(by_difficulty[method].items())
                },
                "by_challenge": {
                    name: metrics(values) for name, values in sorted(by_challenge[method].items())
                },
                "by_category": {
                    name: metrics(values) for name, values in sorted(by_category[method].items())
                },
            }
            for method in methods
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", default=["exact_marker", "marker_family", "exact_surface"],
                        choices=["exact_marker", "marker_family", "exact_surface"])
    args = parser.parse_args()
    report = evaluate(args.pairs.resolve(), args.methods)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)


if __name__ == "__main__":
    main()
