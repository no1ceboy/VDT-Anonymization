"""Build marker-free entity-linking data from reconstructed, name-bearing text.

The anonymized source and reconstruction map provide weak supervision only:
the model input is ``synthetic_markdown`` and the hidden reconstruction
entity_id supplies positive/negative pair labels. Marker strings are never
copied into contexts or handcrafted features.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from itertools import combinations, zip_longest
from pathlib import Path

from .dataset import (
    MENTION_CLOSE,
    MENTION_OPEN,
    allocate_splits,
    cross_entity_pair_candidates,
    normalize_text,
    positive_rank,
    negative_rank,
    row_id,
    stable_hash,
)


RAW_DATASET_VERSION = "entity-linking-raw-v1"
RAW_PAIR_FEATURE_NAMES = (
    "same_label",
    "same_surface",
    "same_role",
    "token_overlap",
    "same_first_token",
    "same_last_token",
    "character_distance_log_scaled",
)


def _apply_replacements(source: str, replacements: list[dict]) -> str:
    result = source
    for item in sorted(replacements, key=lambda value: int(value["start"]), reverse=True):
        start, end = int(item["start"]), int(item["end"])
        result = result[:start] + str(item["replacement"]) + result[end:]
    return result


def _collision_variant(row: dict, mapping: dict, seed: str, collision_rate: float) -> tuple[str, dict, int]:
    """Create a deterministic variant with a shared short person alias.

    Full generated names remain faithful and distinct. Only a short mention
    such as ``An`` is shared, and only when the masked marker initials are
    compatible. This models real ambiguity without inventing an arbitrary full
    identity.
    """
    source = str(row.get("original_anonymized_markdown") or "")
    original_text = str(row.get("synthetic_markdown") or "")
    if collision_rate <= 0:
        return original_text, mapping, 0
    entities = [
        entity for entity in mapping.get("entities", [])
        if entity.get("label") == "PER" and entity.get("synthetic_value")
    ]
    if len(entities) < 2:
        return original_text, mapping, 0
    score = int.from_bytes(hashlib.sha256(f"{seed}|{row_id(row)}|collision".encode()).digest()[:8], "big") / 2**64
    if score >= collision_rate:
        return original_text, mapping, 0
    ordered = sorted(entities, key=lambda entity: stable_hash(row_id(row), entity["entity_id"], seed=seed))

    def marker_initial(entity: dict) -> str:
        match = re.search(r"[A-Za-zÀ-ỹĐđ]", str(entity.get("marker") or ""))
        return match.group(0).casefold() if match else ""

    replacement_items = mapping.get("replacements", [])
    choices = []
    for donor in ordered:
        donor_short = [
            item for item in replacement_items
            if str(item.get("entity_id")) == str(donor["entity_id"])
            and item.get("replacement") is not None
            and len(str(item["replacement"]).strip().split()) == 1
            and len(str(item["replacement"]).strip()) > 1
        ]
        if not donor_short:
            continue
        donor_core = str(donor_short[0]["replacement"]).strip()
        if marker_initial(donor) != donor_core[0].casefold():
            continue
        for recipient in ordered:
            if recipient["entity_id"] == donor["entity_id"] or marker_initial(recipient) != marker_initial(donor):
                continue
            recipient_short = [
                item for item in replacement_items
                if str(item.get("entity_id")) == str(recipient["entity_id"])
                and item.get("replacement") is not None
                and len(str(item["replacement"]).strip().split()) == 1
                and len(str(item["replacement"]).strip()) > 1
            ]
            if recipient_short:
                choices.append((donor, recipient, donor_core))
    if not choices:
        return original_text, mapping, 0
    donor, recipient, donor_core = choices[0]
    recipient_id = str(recipient["entity_id"])
    mutated = copy.deepcopy(mapping)
    collision_count = 0
    for item in mutated.get("replacements", []):
        if str(item.get("entity_id")) != recipient_id or item.get("replacement") is None:
            continue
        current = str(item["replacement"])
        if len(current.strip().split()) != 1:
            continue
        leading = current[:len(current) - len(current.lstrip())]
        trailing = current[len(current.rstrip()):]
        item["replacement"] = leading + donor_core + trailing
        collision_count += 1
    if not collision_count:
        return original_text, mapping, 0
    variant = _apply_replacements(source, mutated["replacements"])
    return variant, mutated, collision_count


def _tokens(value: object) -> list[str]:
    return [token for token in normalize_text(value).replace("-", " ").split() if token]


def _target_replacements(source: str, target: str, mapping: dict) -> list[dict]:
    """Translate source offsets in a reconstruction map to target offsets."""
    replacements = sorted(
        (item for item in mapping.get("replacements", []) if item.get("replacement") is not None),
        key=lambda item: (int(item["start"]), int(item["end"])),
    )
    translated = []
    delta = 0
    previous_end = 0
    for item in replacements:
        start, end = int(item["start"]), int(item["end"])
        if start < previous_end or source[start:end] != item.get("original"):
            raise ValueError("invalid or overlapping reconstruction replacement")
        replacement = str(item["replacement"])
        target_start = start + delta
        target_end = target_start + len(replacement)
        if target[target_start:target_end] != replacement:
            raise ValueError("synthetic text does not match reconstruction map")
        # Replacement planning may add boundary whitespace. Do not include it
        # in the mention surface or in the [MENTION] tags.
        left_trim = len(replacement) - len(replacement.lstrip())
        right_trim = len(replacement.rstrip())
        mention_start = target_start + left_trim
        mention_end = target_start + right_trim
        if mention_start >= mention_end:
            raise ValueError("empty translated mention span")
        translated.append({
            "entity_id": str(item["entity_id"]),
            "label": str(item.get("label") or ""),
            "start": mention_start,
            "end": mention_end,
            "original": target[mention_start:mention_end],
        })
        delta += len(replacement) - (end - start)
        previous_end = end
    if delta != len(target) - len(source):
        raise ValueError("reconstruction map length delta does not match synthetic text")
    return translated


def _mention_context(text: str, item: dict, context_chars: int) -> dict:
    start, end = int(item["start"]), int(item["end"])
    left, right = max(0, start - context_chars), min(len(text), end + context_chars)
    surface = text[start:end]
    context = text[left:start] + MENTION_OPEN + surface + MENTION_CLOSE + text[end:right]
    marked_start = start - left + len(MENTION_OPEN)
    return {
        "label": item["label"],
        "start": start,
        "end": end,
        "surface": surface,
        "context": context,
        "context_mention_start": marked_start,
        "context_mention_end": marked_start + len(surface),
    }


def _same_sentence(text: str, a: dict, b: dict) -> float:
    lo, hi = sorted((int(a["start"]), int(b["start"])))
    return float(not any(mark in text[lo:hi] for mark in (".", "?", "!", "\n\n")))


def raw_pair_features(text: str, a: dict, b: dict) -> dict[str, float]:
    tokens_a, tokens_b = _tokens(a["surface"]), _tokens(b["surface"])
    set_a, set_b = set(tokens_a), set(tokens_b)
    distance = max(0, abs(int(a["start"]) - int(b["start"])))
    return {
        "same_label": float(a.get("label") == b.get("label")),
        "same_surface": float(normalize_text(a.get("surface")) == normalize_text(b.get("surface"))),
        "same_role": float(bool(a.get("role")) and a.get("role") == b.get("role")),
        "token_overlap": float(len(set_a & set_b) / max(1, len(set_a | set_b))),
        "same_first_token": float(bool(tokens_a and tokens_b and tokens_a[0] == tokens_b[0])),
        "same_last_token": float(bool(tokens_a and tokens_b and tokens_a[-1] == tokens_b[-1])),
        "character_distance_log_scaled": min(math.log1p(distance) / 10.0, 1.0),
    }


def _enrich(text: str, item: dict, entity: dict, context_chars: int) -> dict:
    mention = _mention_context(text, item, context_chars)
    mention["role"] = str(entity.get("role") or "")
    return mention


def _make_pair(doc_id: str, split: str, text: str, a: dict, b: dict,
               target: int, difficulty: str, seed: str) -> dict:
    if (a["start"], a["end"]) > (b["start"], b["end"]):
        a, b = b, a
    pair_key = stable_hash(doc_id, a["start"], a["end"], b["start"], b["end"], target, seed=seed)[:20]
    return {
        "pair_id": f"{doc_id}:{pair_key}",
        "split": split,
        "doc_id": doc_id,
        "mention_a": a,
        "mention_b": b,
        "features": raw_pair_features(text, a, b),
        "target_linked": int(target),
        "difficulty": difficulty,
        "supervision": "weak_reconstruction_entity_id",
    }


def build_document_pairs(row: dict, mapping: dict, split: str, context_chars: int,
                         max_positives_per_entity: int, negatives_per_positive: float,
                         min_negatives_per_document: int, max_pairs_per_document: int,
                         seed: str, person_collision_rate: float = 0.30) -> list[dict]:
    source = str(row.get("original_anonymized_markdown") or "")
    text, mapping, _ = _collision_variant(row, mapping, seed, person_collision_rate)
    doc_id = row_id(row)
    entities = {str(entity["entity_id"]): entity for entity in mapping.get("entities", [])}
    translated = _target_replacements(source, text, mapping)
    mentions_by_entity: dict[str, list[dict]] = defaultdict(list)
    for item in translated:
        entity = entities.get(item["entity_id"])
        if entity is None:
            continue
        mentions_by_entity[item["entity_id"]].append(_enrich(text, item, entity, context_chars))

    positives = []
    for entity_id, mentions in sorted(mentions_by_entity.items()):
        candidates = sorted(combinations(mentions, 2), key=lambda p: positive_rank(p[0], p[1], doc_id, seed))
        positives.extend(candidates[:max_positives_per_entity])

    negative_candidates = []
    for entity_a, entity_b in combinations(sorted(mentions_by_entity), 2):
        negative_candidates.extend(cross_entity_pair_candidates(mentions_by_entity[entity_a], mentions_by_entity[entity_b]))
    negative_candidates.sort(key=lambda p: negative_rank(p[0], p[1], doc_id, seed))

    max_positive = max(0, max_pairs_per_document - min_negatives_per_document)
    positives = sorted(positives, key=lambda p: positive_rank(p[0], p[1], doc_id, seed))[:max_positive]
    wanted_negatives = min(
        max(min_negatives_per_document, int(math.ceil(len(positives) * negatives_per_positive))),
        max_pairs_per_document - len(positives),
    )
    negatives = negative_candidates[:wanted_negatives]
    output = [_make_pair(doc_id, split, text, a, b, 1, "positive", seed) for a, b in positives]
    for a, b in negatives:
        same_label = a.get("label") == b.get("label")
        same_surface = normalize_text(a.get("surface")) == normalize_text(b.get("surface"))
        overlap = raw_pair_features(text, a, b)["token_overlap"]
        difficulty = (
            "same_label_surface" if same_label and same_surface else
            "same_label_name_overlap" if same_label and overlap > 0 else
            "same_label" if same_label else "cross_label"
        )
        output.append(_make_pair(doc_id, split, text, a, b, 0, difficulty, seed))
    metadata = {
        "category": str(row.get("category") or "Unknown"),
        "instance_level": str(row.get("instance_level") or "Unknown"),
        "primary_challenge": str((row.get("curation") or {}).get("primary_challenge") or "Unknown"),
    }
    for pair in output:
        pair["document_metadata"] = metadata
    return sorted(output, key=lambda pair: stable_hash(pair["pair_id"], seed=seed))


def load_aligned(clean_path: Path, maps_path: Path) -> list[dict]:
    items, seen_docs, seen_hashes = [], set(), set()
    with clean_path.open(encoding="utf-8-sig") as clean_handle, maps_path.open(encoding="utf-8-sig") as maps_handle:
        for line_number, (clean_line, map_line) in enumerate(zip_longest(clean_handle, maps_handle), 1):
            if clean_line is None or map_line is None:
                raise ValueError(f"input line count mismatch at line {line_number}")
            row, mapping = json.loads(clean_line), json.loads(map_line)
            doc_id = row_id(row)
            source = str(row.get("original_anonymized_markdown") or "")
            source_hash = __import__("hashlib").sha256(source.encode("utf-8")).hexdigest()
            if not doc_id or str(mapping.get("doc_id")) != doc_id or source_hash != mapping.get("source_sha256"):
                raise ValueError(f"source/map mismatch at line {line_number}")
            quality = row.get("reconstruction_quality") or {}
            stats = row.get("reconstruction_stats") or {}
            if not (
                quality.get("eligible") is True
                and quality.get("quality_score") == 100
                and not quality.get("reasons")
                and stats.get("roundtrip_verified") is True
                and int(stats.get("replacements", 0) or 0) > 0
                and str(row.get("synthetic_markdown") or "").strip()
            ):
                raise ValueError(f"input is not strict-clean at line {line_number}: {doc_id}")
            if doc_id in seen_docs or source_hash in seen_hashes:
                raise ValueError(f"document/source leakage candidate at {doc_id}")
            seen_docs.add(doc_id); seen_hashes.add(source_hash)
            items.append({"doc_id": doc_id, "row": row, "map": mapping})
    return items


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_dataset(clean_path: Path, maps_path: Path, output_dir: Path, *,
                  validation_ratio: float = 0.1, test_ratio: float = 0.1,
                  context_chars: int = 160, max_positives_per_entity: int = 8,
                  negatives_per_positive: float = 1.0, min_negatives_per_document: int = 2,
                  max_pairs_per_document: int = 128, seed: str = "vdt-raw-link-v1",
                  person_collision_rate: float = 0.30) -> dict:
    items = load_aligned(clean_path, maps_path)
    assignments = allocate_splits(items, validation_ratio, test_ratio, seed)
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    for child in ("documents", "maps", "pairs"):
        (staging / child).mkdir(parents=True, exist_ok=True)
    handles = {}
    counts = defaultdict(Counter)
    try:
        for split in ("train", "validation", "test"):
            for kind in ("documents", "maps", "pairs"):
                handles[(split, kind)] = (staging / kind / f"{split}.jsonl").open("w", encoding="utf-8")
        for item in sorted(items, key=lambda value: stable_hash(value["doc_id"], seed=seed)):
            split = assignments[item["doc_id"]]
            row, mapping = item["row"], item["map"]
            split_collision_rate = person_collision_rate if split == "train" else 0.0
            input_text, input_mapping, collision_count = _collision_variant(
                row, mapping, seed, split_collision_rate
            )
            document_row = dict(row)
            document_row["entity_linking_input"] = input_text
            document_row["entity_linking_person_collision_replacements"] = collision_count
            handles[(split, "documents")].write(json.dumps(document_row, ensure_ascii=False) + "\n")
            handles[(split, "maps")].write(json.dumps(input_mapping, ensure_ascii=False) + "\n")
            counts[split]["documents"] += 1
            pairs = build_document_pairs(row, mapping, split, context_chars, max_positives_per_entity,
                                         negatives_per_positive, min_negatives_per_document,
                                         max_pairs_per_document, seed, split_collision_rate)
            for pair in pairs:
                handles[(split, "pairs")].write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts[split]["pairs"] += 1
                counts[split]["positive_pairs" if pair["target_linked"] else "negative_pairs"] += 1
                counts[split][f"difficulty:{pair['difficulty']}"] += 1
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "dataset_version": RAW_DATASET_VERSION,
        "task": "within-document entity linking for anonymization",
        "base_model_input": "synthetic_markdown",
        "model_input": "entity_linking_input",
        "synthetic_names_are_pseudonyms": True,
        "supervision": {
            "kind": "weak reconstruction supervision",
            "positive_definition": "two generated-name mentions share a hidden reconstruction entity_id",
            "negative_definition": "two generated-name mentions have different hidden reconstruction entity_id values",
            "marker_in_model_input": False,
            "warning": "Labels originate from the reconstruction linker and are not independent human ground truth.",
        },
        "split_policy": {
            "unit": "document", "stratified_by": ["category", "instance_level", "curation.primary_challenge"],
            "validation_ratio": validation_ratio, "test_ratio": test_ratio, "seed": seed,
            "document_or_source_hash_overlap": 0,
        },
        "pair_policy": {
            "context_chars_each_side": context_chars,
            "max_positives_per_entity": max_positives_per_entity,
            "negatives_per_positive": negatives_per_positive,
            "min_negatives_per_document": min_negatives_per_document,
            "max_pairs_per_document": max_pairs_per_document,
            "feature_names": list(RAW_PAIR_FEATURE_NAMES),
            "person_collision_rate": person_collision_rate,
            "collision_applied_to_splits": ["train"],
            "collision_policy": "training-only augmentation reuses one marker-compatible short PER alias in a subset of documents; full generated names remain distinct and hidden entity_id remains the label",
        },
        "counts": {split: dict(sorted(value.items())) for split, value in sorted(counts.items())},
        "source": {"documents": str(clean_path), "maps": str(maps_path)},
    }
    _save_json(staging / "manifest.json", manifest)
    if output_dir.exists():
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists(): shutil.rmtree(backup)
        os.replace(output_dir, backup); os.replace(staging, output_dir); shutil.rmtree(backup)
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
    parser.add_argument("--person-collision-rate", type=float, default=0.30,
                        help="fraction of eligible documents with an intentional same-name person hard negative")
    parser.add_argument("--seed", default="vdt-raw-link-v1")
    args = parser.parse_args()
    manifest = build_dataset(
        args.clean_input.resolve(), args.maps_input.resolve(), args.output_dir.resolve(),
        validation_ratio=args.validation_ratio, test_ratio=args.test_ratio,
        context_chars=args.context_chars, max_positives_per_entity=args.max_positives_per_entity,
        negatives_per_positive=args.negatives_per_positive,
        min_negatives_per_document=args.min_negatives_per_document,
        max_pairs_per_document=args.max_pairs_per_document, seed=args.seed,
        person_collision_rate=args.person_collision_rate,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
