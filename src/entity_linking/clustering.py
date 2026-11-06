"""Conservative document-level clustering over learned mention-pair scores."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def complete_link_clusters(
    mentions: list[dict],
    pair_scores: dict[tuple[int, int], float],
    threshold: float,
    cannot_link: Iterable[tuple[int, int]] = (),
    must_link: Iterable[tuple[int, int]] = (),
) -> list[list[int]]:
    """Agglomerate mentions using complete-link, with explicit rule constraints.

    Missing scores, incompatible entity labels, and explicit cannot-link pairs
    block a merge. Complete-link (minimum cross score) deliberately avoids the
    classic A~B, B~C, therefore A~C chaining error. High-confidence gazetteer
    matches in ``must_link`` are merged first, unless that would violate a
    cannot-link or entity-type constraint.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    count = len(mentions)
    if not count:
        return []
    blocked_pairs = {_pair_key(int(a), int(b)) for a, b in cannot_link if a != b}
    active = set(range(count))
    members = {index: {index} for index in range(count)}
    linkage: dict[tuple[int, int], float | None] = {}
    blocked: dict[tuple[int, int], bool] = {}
    labels = [str(mention.get("label") or "").strip().casefold() for mention in mentions]

    for left, right in combinations(range(count), 2):
        key = (left, right)
        score = pair_scores.get(key)
        compatible = bool(labels[left]) and labels[left] == labels[right]
        linkage[key] = float(score) if score is not None and compatible else None
        blocked[key] = key in blocked_pairs or not compatible

    def can_merge(left: int, right: int) -> bool:
        return not blocked.get(_pair_key(left, right), True)

    def merge(left: int, right: int) -> None:
        members[left].update(members.pop(right))
        active.remove(right)
        for other in sorted(active - {left}):
            left_key = _pair_key(left, other)
            right_key = _pair_key(right, other)
            left_score, right_score = linkage[left_key], linkage[right_key]
            linkage[left_key] = (
                min(left_score, right_score)
                if left_score is not None and right_score is not None else None
            )
            blocked[left_key] = blocked[left_key] or blocked[right_key]
            linkage.pop(right_key)
            blocked.pop(right_key)
        linkage.pop(_pair_key(left, right), None)
        blocked.pop(_pair_key(left, right), None)

    # Merge only unambiguous same-unit location matches, before model-scored
    # merges. A conflicting parent path always takes precedence.
    def owner(index: int) -> int:
        return next(cluster for cluster in active if index in members[cluster])

    for first, second in sorted({_pair_key(int(a), int(b)) for a, b in must_link if a != b}):
        if not (0 <= first < count and 0 <= second < count):
            continue
        left, right = owner(first), owner(second)
        if left == right or not can_merge(left, right):
            continue
        merge(min(left, right), max(left, right))

    while True:
        candidates = []
        ordered = sorted(active)
        for left, right in combinations(ordered, 2):
            key = (left, right)
            score = linkage[key]
            if not blocked[key] and score is not None and score >= threshold:
                candidates.append((score, left, right))
        if not candidates:
            break
        # Stable tie-break: higher score first, then earlier cluster indices.
        _, left, right = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        merge(left, right)

    return sorted(
        (sorted(members[index]) for index in active),
        key=lambda cluster: min(
            (int(mentions[item].get("start", item)), item) for item in cluster
        ),
    )


def cluster_metrics(gold_entity_ids: list[str], clusters: list[list[int]]) -> dict:
    """Compute B-cubed and pairwise clustering metrics for one document."""
    if len(gold_entity_ids) == 0:
        return {
            "mentions": 0,
            "b_cubed_precision": 0.0,
            "b_cubed_recall": 0.0,
            "b_cubed_f1": 0.0,
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "pairwise_counts": {"tp": 0, "fp": 0, "fn": 0},
        }

    predicted = {}
    for cluster_id, cluster in enumerate(clusters):
        for mention_index in cluster:
            predicted[mention_index] = cluster_id
    if set(predicted) != set(range(len(gold_entity_ids))):
        raise ValueError("clusters must partition every mention index exactly once")

    gold_members: dict[str, set[int]] = {}
    predicted_members: dict[int, set[int]] = {}
    for index, entity_id in enumerate(gold_entity_ids):
        gold_members.setdefault(str(entity_id), set()).add(index)
        predicted_members.setdefault(predicted[index], set()).add(index)

    b_precision = b_recall = 0.0
    for index, entity_id in enumerate(gold_entity_ids):
        intersection = len(gold_members[str(entity_id)] & predicted_members[predicted[index]])
        b_precision += intersection / len(predicted_members[predicted[index]])
        b_recall += intersection / len(gold_members[str(entity_id)])
    b_precision /= len(gold_entity_ids)
    b_recall /= len(gold_entity_ids)
    b_f1 = 2 * b_precision * b_recall / (b_precision + b_recall) if b_precision + b_recall else 0.0

    tp = fp = fn = 0
    for left, right in combinations(range(len(gold_entity_ids)), 2):
        same_gold = str(gold_entity_ids[left]) == str(gold_entity_ids[right])
        same_predicted = predicted[left] == predicted[right]
        if same_gold and same_predicted:
            tp += 1
        elif same_predicted:
            fp += 1
        elif same_gold:
            fn += 1
    pair_precision = tp / (tp + fp) if tp + fp else 0.0
    pair_recall = tp / (tp + fn) if tp + fn else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall else 0.0
    )
    return {
        "mentions": len(gold_entity_ids),
        "b_cubed_precision": b_precision,
        "b_cubed_recall": b_recall,
        "b_cubed_f1": b_f1,
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "pairwise_f1": pair_f1,
        "pairwise_counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def aggregate_cluster_metrics(document_metrics: list[dict]) -> dict:
    """Aggregate B-cubed by mention and pairwise scores by pair counts."""
    mention_count = sum(item["mentions"] for item in document_metrics)
    pair_tp = sum(item["pairwise_counts"]["tp"] for item in document_metrics)
    pair_fp = sum(item["pairwise_counts"]["fp"] for item in document_metrics)
    pair_fn = sum(item["pairwise_counts"]["fn"] for item in document_metrics)
    if mention_count:
        b_precision = sum(item["b_cubed_precision"] * item["mentions"] for item in document_metrics) / mention_count
        b_recall = sum(item["b_cubed_recall"] * item["mentions"] for item in document_metrics) / mention_count
    else:
        b_precision = b_recall = 0.0
    b_f1 = 2 * b_precision * b_recall / (b_precision + b_recall) if b_precision + b_recall else 0.0
    pair_precision = pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else 0.0
    pair_recall = pair_tp / (pair_tp + pair_fn) if pair_tp + pair_fn else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall else 0.0
    )
    return {
        "documents": len(document_metrics),
        "mentions": mention_count,
        "b_cubed": {"precision": b_precision, "recall": b_recall, "f1": b_f1},
        "pairwise": {
            "precision": pair_precision,
            "recall": pair_recall,
            "f1": pair_f1,
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
        },
    }
