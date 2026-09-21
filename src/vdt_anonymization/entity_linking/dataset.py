"""Build document-disjoint weakly supervised entity-linking datasets.

The reconstruction maps are treated as weak ground truth: two mention spans are
linked when the production linker assigned them the same entity_id.  Synthetic
replacement values are deliberately never copied into pair examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import combinations, zip_longest
from pathlib import Path


DATASET_VERSION = "entity-linking-weak-v1"
MENTION_OPEN = "[MENTION]"
MENTION_CLOSE = "[/MENTION]"
PAIR_FEATURE_NAMES = (
    "same_label",
    "same_surface",
    "same_marker",
    "same_marker_family",
    "same_role",
    "both_numbered",
    "number_suffix_gap_scaled",
    "character_distance_log_scaled",
)


def stable_hash(*parts: object, seed: str = "vdt-entity-linking-v1") -> str:
    payload = "\x1f".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    return " ".join(text.split())


def marker_family(value: object) -> str:
    return re.sub(r"\d+$", "", normalize_text(value)).strip()


def number_suffix(value: object) -> int | None:
    match = re.search(r"(\d+)$", normalize_text(value))
    return int(match.group(1)) if match else None


def row_id(row: dict) -> str:
    for field in ("doc_name", "case_id", "official_document_id", "id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def stratum(row: dict) -> str:
    curation = row.get("curation") or {}
    return "|".join(
        str(value or "Unknown")
        for value in (
            row.get("category"),
            row.get("instance_level"),
            curation.get("primary_challenge"),
        )
    )


def allocate_splits(rows: list[dict], validation_ratio: float, test_ratio: float,
                    seed: str) -> dict[str, str]:
    """Assign complete documents to deterministic, approximately stratified splits."""
    if not 0 <= validation_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("validation and test ratios must be in [0, 1)")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        groups[stratum(item["row"])].append(item)

    assignments: dict[str, str] = {}
    for group_name, items in sorted(groups.items()):
        ranked = sorted(items, key=lambda item: stable_hash(group_name, item["doc_id"], seed=seed))
        count = len(ranked)
        val_count = int(math.floor(count * validation_ratio))
        test_count = int(math.floor(count * test_ratio))
        # Preserve representation for meaningful strata without taking most of a tiny group.
        if count >= 10 and validation_ratio > 0:
            val_count = max(1, val_count)
        if count >= 10 and test_ratio > 0:
            test_count = max(1, test_count)
        for index, item in enumerate(ranked):
            split = "validation" if index < val_count else "test" if index < val_count + test_count else "train"
            assignments[item["doc_id"]] = split

    # Guarantee global held-out coverage for rare challenge families whenever
    # three documents exist, even when category/instance subdivision is tiny.
    by_challenge: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        challenge = str((item["row"].get("curation") or {}).get("primary_challenge") or "Unknown")
        by_challenge[challenge].append(item)
    for challenge, items in sorted(by_challenge.items()):
        if len(items) < 3:
            continue
        ranked = sorted(items, key=lambda item: stable_hash("challenge", challenge, item["doc_id"], seed=seed))
        present = {assignments[item["doc_id"]] for item in ranked}
        for required in ("validation", "test"):
            if required in present:
                continue
            donor = next((item for item in ranked if assignments[item["doc_id"]] == "train"), None)
            if donor is not None:
                assignments[donor["doc_id"]] = required
                present.add(required)
    return assignments


def mention_context(source: str, replacement: dict, context_chars: int) -> dict:
    start, end = int(replacement["start"]), int(replacement["end"])
    left = max(0, start - context_chars)
    right = min(len(source), end + context_chars)
    surface = source[start:end]
    context = source[left:start] + MENTION_OPEN + surface + MENTION_CLOSE + source[end:right]
    marked_start = start - left + len(MENTION_OPEN)
    return {
        "entity_id": str(replacement["entity_id"]),
        "label": str(replacement.get("label") or ""),
        "start": start,
        "end": end,
        "surface": surface,
        "context": context,
        "context_mention_start": marked_start,
        "context_mention_end": marked_start + len(surface),
    }


def enrich_mention(mention: dict, entity: dict) -> dict:
    result = dict(mention)
    result.update({
        "marker": str(entity.get("marker") or ""),
        "marker_family": marker_family(entity.get("marker")),
        "role": str(entity.get("role") or ""),
    })
    return result


def pair_features(a: dict, b: dict) -> dict[str, float]:
    suffix_a, suffix_b = number_suffix(a.get("marker")), number_suffix(b.get("marker"))
    distance = max(0, abs(int(a["start"]) - int(b["start"])))
    normalized_marker_a, normalized_marker_b = normalize_text(a.get("marker")), normalize_text(b.get("marker"))
    return {
        "same_label": float(a.get("label") == b.get("label")),
        "same_surface": float(normalize_text(a.get("surface")) == normalize_text(b.get("surface"))),
        "same_marker": float(bool(normalized_marker_a) and normalized_marker_a == normalized_marker_b),
        "same_marker_family": float(
            bool(a.get("marker_family")) and a.get("marker_family") == b.get("marker_family")
        ),
        "same_role": float(bool(a.get("role")) and a.get("role") == b.get("role")),
        "both_numbered": float(suffix_a is not None and suffix_b is not None),
        "number_suffix_gap_scaled": (
            min(abs(suffix_a - suffix_b), 20) / 20.0
            if suffix_a is not None and suffix_b is not None else 0.0
        ),
        "character_distance_log_scaled": min(math.log1p(distance) / 10.0, 1.0),
    }


def positive_rank(a: dict, b: dict, doc_id: str, seed: str) -> tuple:
    # Prefer a full/long mention paired with a short reference, then nearby mentions.
    length_gap = abs(len(a["surface"]) - len(b["surface"]))
    different_surface = normalize_text(a["surface"]) != normalize_text(b["surface"])
    distance = abs(a["start"] - b["start"])
    return (-int(different_surface), -length_gap, distance, stable_hash(doc_id, a["start"], b["start"], seed=seed))


def negative_rank(a: dict, b: dict, doc_id: str, seed: str) -> tuple:
    same_label = a.get("label") == b.get("label")
    same_marker = normalize_text(a.get("marker")) == normalize_text(b.get("marker"))
    same_family = bool(a.get("marker_family")) and a.get("marker_family") == b.get("marker_family")
    same_surface = normalize_text(a.get("surface")) == normalize_text(b.get("surface"))
    priority = (
        0 if same_label and same_surface else
        1 if same_label and same_marker else
        2 if same_label and same_family else
        3 if same_label else 4
    )
    # Distance is deliberately excluded from this key. An ascending-distance
    # tiebreak here always selected the closest available candidate first,
    # while same-entity positive pairs (see positive_rank) span the whole
    # document. That taught "far apart" as a proxy for "same person" -- a
    # shortcut with nothing to do with coreference. A seeded hash keeps
    # selection deterministic without biasing toward any distance.
    return (priority, stable_hash(doc_id, a["start"], b["start"], seed=seed))


def nearby_cross_entity_pairs(mentions_a: list[dict], mentions_b: list[dict], limit: int = 4) -> list[tuple[dict, dict]]:
    """Return a bounded set of nearby pairs without a quadratic cross product."""
    if len(mentions_a) > len(mentions_b):
        return [(b, a) for a, b in nearby_cross_entity_pairs(mentions_b, mentions_a, limit)]
    ordered_b = sorted(mentions_b, key=lambda mention: mention["start"])
    positions_b = [mention["start"] for mention in ordered_b]
    candidates = {}
    for mention_a in mentions_a:
        insertion = bisect_left(positions_b, mention_a["start"])
        for index in (insertion - 1, insertion, insertion + 1):
            if 0 <= index < len(ordered_b):
                mention_b = ordered_b[index]
                key = (mention_a["start"], mention_a["end"], mention_b["start"], mention_b["end"])
                candidates[key] = (mention_a, mention_b)
    return sorted(candidates.values(), key=lambda pair: abs(pair[0]["start"] - pair[1]["start"]))[:limit]


def cross_entity_pair_candidates(mentions_a: list[dict], mentions_b: list[dict], limit: int = 4) -> list[tuple[dict, dict]]:
    """Distance-diverse cross-entity negative candidates.

    ``nearby_cross_entity_pairs`` alone only ever returns adjacent mentions,
    so every cross-entity negative sits close together while a single
    entity's own mentions (positives) can span the entire document. A model
    can then reach high accuracy by learning "far apart" implies "same
    person" instead of reading either mention. This keeps the nearby (hard,
    locally-confusable) candidates and adds the single most distant
    cross-entity pair, so distant negatives exist for training to see too.
    """
    nearby_limit = max(1, limit - 1)
    nearby = nearby_cross_entity_pairs(mentions_a, mentions_b, nearby_limit)
    candidates = {
        (a["start"], a["end"], b["start"], b["end"]): (a, b) for a, b in nearby
    }
    first_a, last_a = min(mentions_a, key=lambda m: m["start"]), max(mentions_a, key=lambda m: m["start"])
    first_b, last_b = min(mentions_b, key=lambda m: m["start"]), max(mentions_b, key=lambda m: m["start"])
    farthest = max(
        ((first_a, last_b), (last_a, first_b)),
        key=lambda pair: abs(pair[0]["start"] - pair[1]["start"]),
    )
    candidates[(farthest[0]["start"], farthest[0]["end"], farthest[1]["start"], farthest[1]["end"])] = farthest
    return list(candidates.values())[:limit]


def make_pair(doc_id: str, split: str, a: dict, b: dict, target: int,
              difficulty: str, seed: str) -> dict:
    if (a["start"], a["end"]) > (b["start"], b["end"]):
        a, b = b, a
    pair_key = stable_hash(doc_id, a["start"], a["end"], b["start"], b["end"], target, seed=seed)[:20]
    return {
        "pair_id": f"{doc_id}:{pair_key}",
        "split": split,
        "doc_id": doc_id,
        "mention_a": a,
        "mention_b": b,
        "features": pair_features(a, b),
        "target_linked": target,
        "difficulty": difficulty,
        "supervision": "weak_rule_entity_id",
    }


def build_document_pairs(row: dict, mapping: dict, split: str, context_chars: int,
                         max_positives_per_entity: int, negatives_per_positive: float,
                         min_negatives_per_document: int, max_pairs_per_document: int,
                         seed: str) -> list[dict]:
    source = str(row.get("original_anonymized_markdown") or "")
    doc_id = row_id(row)
    entities = {str(entity["entity_id"]): entity for entity in mapping.get("entities", [])}
    mentions_by_entity: dict[str, list[dict]] = defaultdict(list)
    for replacement in mapping.get("replacements", []):
        entity_id = str(replacement.get("entity_id") or "")
        entity = entities.get(entity_id)
        start, end = replacement.get("start"), replacement.get("end")
        if entity is None or not isinstance(start, int) or not isinstance(end, int):
            continue
        if not (0 <= start < end <= len(source)) or source[start:end] != replacement.get("original"):
            continue
        mentions_by_entity[entity_id].append(
            enrich_mention(mention_context(source, replacement, context_chars), entity)
        )

    positives: list[tuple[dict, dict]] = []
    for entity_id, mentions in sorted(mentions_by_entity.items()):
        candidates = sorted(
            combinations(mentions, 2),
            key=lambda pair: positive_rank(pair[0], pair[1], doc_id, seed),
        )
        positives.extend(candidates[:max_positives_per_entity])

    negative_candidates: list[tuple[dict, dict]] = []
    for entity_a, entity_b in combinations(sorted(mentions_by_entity), 2):
        negative_candidates.extend(
            cross_entity_pair_candidates(mentions_by_entity[entity_a], mentions_by_entity[entity_b])
        )
    negative_candidates.sort(key=lambda pair: negative_rank(pair[0], pair[1], doc_id, seed))

    max_positive = max(0, max_pairs_per_document - min_negatives_per_document)
    positives = sorted(positives, key=lambda pair: positive_rank(pair[0], pair[1], doc_id, seed))[:max_positive]
    wanted_negatives = max(min_negatives_per_document, int(math.ceil(len(positives) * negatives_per_positive)))
    wanted_negatives = min(wanted_negatives, max_pairs_per_document - len(positives))
    negatives = negative_candidates[:wanted_negatives]

    output = [make_pair(doc_id, split, a, b, 1, "positive", seed) for a, b in positives]
    for a, b in negatives:
        rank = negative_rank(a, b, doc_id, seed)[0]
        difficulty = {0: "same_label_surface", 1: "same_label_marker", 2: "same_label_marker_family", 3: "same_label"}.get(rank, "cross_label")
        output.append(make_pair(doc_id, split, a, b, 0, difficulty, seed))
    metadata = {
        "category": str(row.get("category") or "Unknown"),
        "instance_level": str(row.get("instance_level") or "Unknown"),
        "primary_challenge": str((row.get("curation") or {}).get("primary_challenge") or "Unknown"),
    }
    for pair in output:
        pair["document_metadata"] = metadata
    return sorted(output, key=lambda pair: stable_hash(pair["pair_id"], seed=seed))


def load_aligned(clean_path: Path, maps_path: Path) -> list[dict]:
    items = []
    seen_docs, seen_hashes = set(), set()
    with clean_path.open(encoding="utf-8-sig") as clean_handle, maps_path.open(encoding="utf-8-sig") as maps_handle:
        for line_number, (clean_line, map_line) in enumerate(zip_longest(clean_handle, maps_handle), 1):
            if clean_line is None or map_line is None:
                raise ValueError(f"input line count mismatch at line {line_number}")
            row, mapping = json.loads(clean_line), json.loads(map_line)
            doc_id = row_id(row)
            source = str(row.get("original_anonymized_markdown") or "")
            source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if not doc_id or str(mapping.get("doc_id")) != doc_id:
                raise ValueError(f"map ID mismatch at line {line_number}")
            if source_hash != mapping.get("source_sha256"):
                raise ValueError(f"source hash mismatch for {doc_id}")
            if doc_id in seen_docs or source_hash in seen_hashes:
                raise ValueError(f"document/source leakage candidate at {doc_id}")
            seen_docs.add(doc_id)
            seen_hashes.add(source_hash)
            items.append({"doc_id": doc_id, "source_hash": source_hash, "row": row, "map": mapping})
    return items


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_dataset(clean_path: Path, maps_path: Path, output_dir: Path, *,
                  validation_ratio: float = 0.1, test_ratio: float = 0.1,
                  context_chars: int = 160, max_positives_per_entity: int = 8,
                  negatives_per_positive: float = 1.0, min_negatives_per_document: int = 2,
                  max_pairs_per_document: int = 128, seed: str = "vdt-link-v1") -> dict:
    items = load_aligned(clean_path, maps_path)
    assignments = allocate_splits(items, validation_ratio, test_ratio, seed)
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    for child in ("documents", "maps", "pairs"):
        (staging / child).mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = defaultdict(Counter)
    challenge_counts = defaultdict(Counter)
    try:
        for split in ("train", "validation", "test"):
            handles[(split, "documents")] = (staging / "documents" / f"{split}.jsonl").open("w", encoding="utf-8")
            handles[(split, "maps")] = (staging / "maps" / f"{split}.jsonl").open("w", encoding="utf-8")
            handles[(split, "pairs")] = (staging / "pairs" / f"{split}.jsonl").open("w", encoding="utf-8")

        for item in sorted(items, key=lambda value: stable_hash(value["doc_id"], seed=seed)):
            split = assignments[item["doc_id"]]
            row, mapping = item["row"], item["map"]
            handles[(split, "documents")].write(json.dumps(row, ensure_ascii=False) + "\n")
            handles[(split, "maps")].write(json.dumps(mapping, ensure_ascii=False) + "\n")
            counts[split]["documents"] += 1
            challenge = str((row.get("curation") or {}).get("primary_challenge") or "Unknown")
            challenge_counts[split][challenge] += 1
            pairs = build_document_pairs(
                row, mapping, split, context_chars, max_positives_per_entity,
                negatives_per_positive, min_negatives_per_document,
                max_pairs_per_document, seed,
            )
            for pair in pairs:
                handles[(split, "pairs")].write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts[split]["pairs"] += 1
                counts[split]["positive_pairs" if pair["target_linked"] else "negative_pairs"] += 1
                counts[split][f"difficulty:{pair['difficulty']}"] += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "dataset_version": DATASET_VERSION,
        "supervision": {
            "kind": "weak supervision",
            "positive_definition": "distinct source spans assigned the same reconstruction entity_id",
            "negative_definition": "source spans assigned different entity_id values in the same document",
            "warning": "Labels are produced by the rule-based reconstruction linker, not independently human-annotated ground truth.",
            "synthetic_values_in_model_input": False,
        },
        "split_policy": {
            "unit": "document",
            "stratified_by": ["category", "instance_level", "curation.primary_challenge"],
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "document_or_source_hash_overlap": 0,
        },
        "pair_policy": {
            "context_chars_each_side": context_chars,
            "max_positives_per_entity": max_positives_per_entity,
            "negatives_per_positive": negatives_per_positive,
            "min_negatives_per_document": min_negatives_per_document,
            "max_pairs_per_document": max_pairs_per_document,
            "hard_negative_order": ["same_label_surface", "same_label_marker", "same_label_marker_family", "same_label", "cross_label"],
            "feature_names": list(PAIR_FEATURE_NAMES),
        },
        "counts": {split: dict(sorted(value.items())) for split, value in sorted(counts.items())},
        "challenge_distribution": {
            split: dict(sorted(value.items())) for split, value in sorted(challenge_counts.items())
        },
        "source": {"documents": str(clean_path), "maps": str(maps_path)},
    }
    atomic_json(staging / "manifest.json", manifest)
    if output_dir.exists():
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output_dir, backup)
        os.replace(staging, output_dir)
        shutil.rmtree(backup)
    else:
        os.replace(staging, output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-input", required=True, type=Path)
    parser.add_argument("--maps-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--max-positives-per-entity", type=int, default=8)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--min-negatives-per-document", type=int, default=2)
    parser.add_argument("--max-pairs-per-document", type=int, default=128)
    parser.add_argument("--seed", default="vdt-link-v1")
    args = parser.parse_args()
    if args.context_chars < 16 or args.max_pairs_per_document < 2:
        parser.error("context must be >=16 characters and max pairs must be >=2")
    manifest = build_dataset(
        args.clean_input.resolve(), args.maps_input.resolve(), args.output_dir.resolve(),
        validation_ratio=args.validation_ratio, test_ratio=args.test_ratio,
        context_chars=args.context_chars, max_positives_per_entity=args.max_positives_per_entity,
        negatives_per_positive=args.negatives_per_positive,
        min_negatives_per_document=args.min_negatives_per_document,
        max_pairs_per_document=args.max_pairs_per_document, seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
