"""Evaluate the normalized-text and location-rule baseline against pair labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import normalize_text
from .location_constraints import location_rule_decision


BASELINE_NAME = "normalized_surface_plus_location_rules"


def predict(pair: dict) -> tuple[int, str | None]:
    """Apply high-confidence location rules, then same-label exact text matching."""
    location_decision = location_rule_decision(pair)
    if location_decision == "block":
        return 0, location_decision
    if location_decision == "link":
        return 1, location_decision

    a, b = pair["mention_a"], pair["mention_b"]
    same_label = a.get("label") == b.get("label")
    same_surface = normalize_text(a.get("surface")) == normalize_text(b.get("surface"))
    return int(same_label and same_surface), location_decision


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


def _metrics_by_group(groups: dict[str, Counter]) -> dict:
    return {name: metrics(values) for name, values in sorted(groups.items())}


def evaluate(path: Path) -> dict:
    total = Counter()
    by_difficulty: dict[str, Counter] = defaultdict(Counter)
    by_challenge: dict[str, Counter] = defaultdict(Counter)
    by_category: dict[str, Counter] = defaultdict(Counter)
    location_rules = Counter()
    pair_count = 0

    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            pair_count += 1
            target = int(pair["target_linked"])
            prediction, location_decision = predict(pair)
            outcome = "tp" if target and prediction else "fn" if target else "fp" if prediction else "tn"
            total[outcome] += 1
            by_difficulty[str(pair.get("difficulty") or "Unknown")][outcome] += 1

            metadata = pair.get("document_metadata") or {}
            by_challenge[str(metadata.get("primary_challenge") or "Unknown")][outcome] += 1
            by_category[str(metadata.get("category") or "Unknown")][outcome] += 1

            if location_decision:
                location_rules[location_decision] += 1
                if location_decision == "block" and target:
                    location_rules["weak_label_disagreements"] += 1
                elif location_decision == "link" and not target:
                    location_rules["weak_label_disagreements"] += 1

    if not pair_count:
        raise ValueError(f"No pairs found in {path}")

    return {
        "evaluation_kind": "agreement with weak rule-generated labels; not independent human-ground-truth accuracy",
        "baseline": BASELINE_NAME,
        "rule": (
            "For LOC, follow a high-confidence canonical administrative-unit match or conflict; "
            "otherwise link only if NER labels and normalized mention surfaces match."
        ),
        "pair_file": str(path),
        "metrics": {
            "overall": metrics(total),
            "by_difficulty": _metrics_by_group(by_difficulty),
            "by_primary_challenge": _metrics_by_group(by_challenge),
            "by_document_category": _metrics_by_group(by_category),
        },
        "location_rule_counts": {
            "total_pairs": pair_count,
            "decided_pairs": location_rules["link"] + location_rules["block"],
            "links": location_rules["link"],
            "blocks": location_rules["block"],
            "deferred_pairs": pair_count - location_rules["link"] - location_rules["block"],
            "weak_label_disagreements": location_rules["weak_label_disagreements"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.pairs.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)


if __name__ == "__main__":
    main()
