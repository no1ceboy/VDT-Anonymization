"""Build V4 entity-linking data from original anonymized documents.

V4 reuses V3's document-level splits and weak reconstruction entity IDs, but
does not use V3's generated-name text.  Its additional person-collision cases
are deterministic counterfactuals: one short PER alias is replaced with a
different PER entity's existing alias, while the original entity IDs remain
the labels.  They are kept in separate validation/test challenge files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from collections import Counter, defaultdict
from itertools import combinations, zip_longest
from pathlib import Path

from .dataset import MENTION_CLOSE, MENTION_OPEN, normalize_text, stable_hash
from .location_constraints import location_rule_decision
from .raw_dataset import RAW_PAIR_FEATURE_NAMES, cross_entity_pair_candidates, raw_pair_features

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


DATASET_VERSION = "entity-linking-original-anonymized-v4-location-hierarchy"
SPLITS = ("train", "validation", "test")
_MARKER = re.compile(r"^(?:[^\W_]{1,3})$", re.UNICODE)
def _doc_id(row: dict) -> str:
    return str(row.get("doc_name") or row.get("case_id") or row.get("id") or "").strip()


def _load_split(root: Path, split: str) -> Iterator[dict]:
    documents = root / "documents" / f"{split}.jsonl"
    maps = root / "maps" / f"{split}.jsonl"
    if not documents.is_file() or not maps.is_file():
        raise FileNotFoundError(f"V3 {split} documents/maps are missing under {root}")
    seen_ids = set()
    with documents.open(encoding="utf-8-sig") as document_file, maps.open(encoding="utf-8-sig") as map_file:
        for line_number, (document_line, map_line) in enumerate(zip_longest(document_file, map_file), 1):
            if document_line is None or map_line is None:
                raise ValueError(f"V3 {split} document/map count mismatch at line {line_number}")
            row, mapping = json.loads(document_line), json.loads(map_line)
            doc_id = _doc_id(row)
            text = str(row.get("original_anonymized_markdown") or "")
            if not doc_id or str(mapping.get("doc_id") or "") != doc_id:
                raise ValueError(f"V3 {split} document/map ID mismatch at line {line_number}")
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if not text or source_hash != mapping.get("source_sha256"):
                raise ValueError(f"V3 source/map hash mismatch for {doc_id}")
            if doc_id in seen_ids:
                raise ValueError(f"duplicate document ID {doc_id} in V3 {split}")
            seen_ids.add(doc_id)
            for replacement in mapping.get("replacements", []):
                start, end = replacement.get("start"), replacement.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(f"invalid source offsets for {doc_id}")
                if not (0 <= start < end <= len(text)) or text[start:end] != replacement.get("original"):
                    raise ValueError(f"V3 source span mismatch for {doc_id} at [{start}, {end})")
            yield {"doc_id": doc_id, "row": row, "map": mapping, "text": text}


def _metadata(row: dict) -> dict:
    curation = row.get("curation") or {}
    return {
        "category": str(row.get("category") or "Unknown"),
        "instance_level": str(row.get("instance_level") or "Unknown"),
        "primary_challenge": str(curation.get("primary_challenge") or "Unknown"),
    }


def _clean_document(row: dict, text: str, variant: str = "original_anonymized") -> dict:
    """Allowlist metadata; generated names and reconstruction diagnostics never pass through."""
    keys = (
        "source",
        "web_url",
        "official_document_id",
        "official_document_id_normalized",
        "number",
        "year",
        "category",
        "instance_level",
        "court",
        "court_level",
        "issued_date",
    )
    output = {key: row[key] for key in keys if row.get(key) is not None}
    output.update({
        "doc_name": _doc_id(row),
        "curation": {"primary_challenge": _metadata(row)["primary_challenge"]},
        "original_anonymized_markdown": text,
        "entity_linking_input": text,
        "input_variant": variant,
    })
    return output


def _clean_map(mapping: dict, text: str, *, base_source_sha256: str | None = None) -> dict:
    """Keep only weak identity IDs, labels, and source-aligned spans."""
    entities = [
        {"entity_id": str(entity["entity_id"]), "label": str(entity.get("label") or "")}
        for entity in mapping.get("entities", [])
        if entity.get("entity_id") and entity.get("label")
    ]
    replacements = []
    for item in mapping.get("replacements", []):
        start, end = int(item["start"]), int(item["end"])
        surface = text[start:end]
        if surface != item.get("original"):
            raise ValueError(f"map span mismatch at [{start}, {end})")
        replacements.append({
            "entity_id": str(item["entity_id"]),
            "label": str(item.get("label") or ""),
            "start": start,
            "end": end,
            "original": surface,
            # The document evaluator translates maps through this field.  For
            # V4 it is intentionally an identity mapping over anonymized text.
            "replacement": surface,
        })
    output = {
        "doc_id": str(mapping["doc_id"]),
        "pipeline_version": "v4-original-anonymized-weak-map",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entities": entities,
        "replacements": replacements,
    }
    if base_source_sha256:
        output["base_source_sha256"] = base_source_sha256
    return output


def _mention(text: str, item: dict, context_chars: int) -> dict:
    start, end = int(item["start"]), int(item["end"])
    left, right = max(0, start - context_chars), min(len(text), end + context_chars)
    surface = text[start:end]
    context = text[left:start] + MENTION_OPEN + surface + MENTION_CLOSE + text[end:right]
    mention_start = start - left + len(MENTION_OPEN)
    return {
        "label": str(item.get("label") or ""),
        "start": start,
        "end": end,
        "surface": surface,
        "context": context,
        "context_mention_start": mention_start,
        "context_mention_end": mention_start + len(surface),
    }


def _pair(
    doc_id: str,
    split: str,
    text: str,
    a: dict,
    b: dict,
    target: int,
    difficulty: str,
    seed: str,
    variant: str,
    metadata: dict,
) -> dict:
    if (a["start"], a["end"]) > (b["start"], b["end"]):
        a, b = b, a
    key = stable_hash(doc_id, variant, a["start"], a["end"], b["start"], b["end"], target, seed=seed)[:20]
    return {
        "pair_id": f"{doc_id}:{key}",
        "split": split,
        "doc_id": doc_id,
        "mention_a": a,
        "mention_b": b,
        "features": raw_pair_features(text, a, b),
        "target_linked": int(target),
        "difficulty": difficulty,
        "supervision": "weak_reconstruction_entity_id",
        "input_variant": variant,
        "document_metadata": metadata,
    }


def _pair_key(a: dict, b: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    first, second = (a, b) if (a["start"], a["end"]) <= (b["start"], b["end"]) else (b, a)
    return (first["start"], first["end"]), (second["start"], second["end"])


def build_natural_pairs(
    doc_id: str,
    split: str,
    text: str,
    mapping: dict,
    metadata: dict,
    *,
    context_chars: int = 160,
    max_positives_per_entity: int = 8,
    max_pairs_per_document: int = 128,
    seed: str = "vdt-link-v4",
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Create balanced pairs with guaranteed natural same-surface negatives.

    Negatives are prioritized by exact surface, conflicting location hierarchy,
    same-label token overlap, then remaining same-/cross-label candidates.
    Near and far pairs are both present to reduce distance shortcuts.
    """
    entities = {str(entity["entity_id"]): entity for entity in mapping.get("entities", [])}
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for item in mapping.get("replacements", []):
        entity_id = str(item.get("entity_id") or "")
        entity = entities.get(entity_id)
        if entity is None:
            continue
        mention = _mention(text, item, context_chars)
        if not mention["label"]:
            mention["label"] = str(entity.get("label") or "")
        if mention["label"]:
            by_entity[entity_id].append(mention)

    positives = []
    for entity_id, mentions in sorted(by_entity.items()):
        candidates = list(combinations(mentions, 2))
        candidates.sort(key=lambda pair: (
            -int(normalize_text(pair[0]["surface"]) != normalize_text(pair[1]["surface"])),
            -abs(len(pair[0]["surface"]) - len(pair[1]["surface"])),
            abs(pair[0]["start"] - pair[1]["start"]),
            stable_hash(doc_id, entity_id, pair[0]["start"], pair[1]["start"], seed=seed),
        ))
        positives.extend(candidates[:max_positives_per_entity])

    # Do not ask the encoder to learn a pair label that contradicts a
    # high-confidence canonical location decision. These are omitted rather
    # than silently relabeled, keeping the weak-map source explicit.
    rule_exclusions: Counter = Counter()
    consistent_positives = []
    for a, b in positives:
        decision = location_rule_decision({"mention_a": a, "mention_b": b})
        if decision == "block":
            rule_exclusions["weak_positive_blocked_by_location_rule"] += 1
        else:
            consistent_positives.append((a, b))
    positives = consistent_positives

    negatives: dict[tuple, tuple[dict, dict]] = {}

    def add_negative(a: dict, b: dict) -> None:
        if a["label"].casefold() == b["label"].casefold() and a["start"] != b["start"]:
            negatives[_pair_key(a, b)] = (a, b)

    # Guaranteed exact-surface hard negatives, even when they are far apart.
    by_surface: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for entity_id, mentions in by_entity.items():
        for mention in mentions:
            by_surface[(mention["label"].casefold(), normalize_text(mention["surface"]))].append((entity_id, mention))
    for occurrences in by_surface.values():
        for (id_a, a), (id_b, b) in combinations(occurrences, 2):
            if id_a != id_b:
                add_negative(a, b)

    # Add local and long-distance examples across different entities.
    for entity_a, entity_b in combinations(sorted(by_entity), 2):
        for a, b in cross_entity_pair_candidates(by_entity[entity_a], by_entity[entity_b]):
            add_negative(a, b)

    def negative_rank(pair: tuple[dict, dict]) -> tuple:
        a, b = pair
        features = raw_pair_features(text, a, b)
        same_surface = normalize_text(a["surface"]) == normalize_text(b["surface"])
        overlap = features["token_overlap"]
        priority = (
            0 if same_surface else
            1 if location_rule_decision({"mention_a": a, "mention_b": b}) == "block" else
            2 if overlap > 0 else 3
        )
        return (
            priority,
            stable_hash(doc_id, a["start"], b["start"], seed=seed),
        )

    consistent_negatives = {}
    for key, pair in negatives.items():
        if location_rule_decision({"mention_a": pair[0], "mention_b": pair[1]}) == "link":
            rule_exclusions["weak_negative_linked_by_location_rule"] += 1
        else:
            consistent_negatives[key] = pair
    negatives = consistent_negatives
    ordered_negatives = sorted(negatives.values(), key=negative_rank)
    cap = max(2, max_pairs_per_document)
    positive_quota = cap // 2
    negative_quota = cap - positive_quota
    ordered_positives = sorted(positives, key=lambda pair: (
        stable_hash(doc_id, pair[0]["start"], pair[1]["start"], seed=seed)
    ))
    selected_pos = ordered_positives[:positive_quota]
    selected_neg = ordered_negatives[:negative_quota]
    remaining = cap - len(selected_pos) - len(selected_neg)
    extra_pos = iter(ordered_positives[len(selected_pos):])
    extra_neg = iter(ordered_negatives[len(selected_neg):])
    while remaining > 0:
        if len(selected_pos) <= len(selected_neg):
            next_pair = next(extra_pos, None)
            if next_pair is not None:
                selected_pos.append(next_pair)
                remaining -= 1
                continue
        next_pair = next(extra_neg, None)
        if next_pair is not None:
            selected_neg.append(next_pair)
            remaining -= 1
            continue
        next_pair = next(extra_pos, None)
        if next_pair is None:
            break
        selected_pos.append(next_pair)
        remaining -= 1

    output = [
        _pair(doc_id, split, text, a, b, 1, "positive", seed, "original_anonymized", metadata)
        for a, b in selected_pos
    ]
    for a, b in selected_neg:
        features = raw_pair_features(text, a, b)
        difficulty = (
            "same_label_surface" if normalize_text(a["surface"]) == normalize_text(b["surface"]) else
            "same_label_location_conflict"
            if location_rule_decision({"mention_a": a, "mention_b": b}) == "block" else
            "same_label_name_overlap" if features["token_overlap"] > 0 else
            "same_label"
        )
        output.append(_pair(doc_id, split, text, a, b, 0, difficulty, seed, "original_anonymized", metadata))

    # Separate natural location diagnostics are intentionally untruncated.
    surface_hard = []
    location_hard = []
    for (label, surface), occurrences in by_surface.items():
        for (id_a, a), (id_b, b) in combinations(occurrences, 2):
            if id_a == id_b:
                continue
            if (label == a["label"].casefold() and label == b["label"].casefold()
                    and location_rule_decision({"mention_a": a, "mention_b": b}) != "link"):
                surface_hard.append(_pair(
                    doc_id, split, text, a, b, 0, "same_label_surface", seed,
                    "original_anonymized", metadata,
                ))
    # Keep a separate untruncated location diagnostic file, covering conflicts
    # at province, district, and commune levels. Include sampled hard negatives
    # beyond exact-surface collisions when they carry a contradictory path.
    for a, b in ordered_negatives:
        if location_rule_decision({"mention_a": a, "mention_b": b}) == "block":
            location_hard.append(_pair(
                doc_id, split, text, a, b, 0, "location_hierarchy_conflict", seed,
                "original_anonymized", metadata,
            ))
    return output, surface_hard + location_hard, dict(rule_exclusions)


def _marker_like(value: str) -> bool:
    value = str(value or "").strip()
    if not value or len(value) > 3 or len(value.split()) != 1 or not _MARKER.fullmatch(value):
        return False
    letters = value.rstrip("0123456789")
    return bool(letters) and (
        letters.isupper() or (letters[0].isupper() and letters[1:].islower())
    )


def _collision_candidates(item: dict) -> list[dict]:
    text, mapping, doc_id = item["text"], item["map"], item["doc_id"]
    entity_rows = {str(e["entity_id"]): e for e in mapping.get("entities", []) if e.get("label") == "PER"}
    mentions: dict[str, list[dict]] = defaultdict(list)
    for replacement in mapping.get("replacements", []):
        entity_id = str(replacement.get("entity_id") or "")
        if entity_id not in entity_rows or replacement.get("label") != "PER":
            continue
        surface = text[int(replacement["start"]):int(replacement["end"])]
        mentions[entity_id].append({**replacement, "surface": surface})

    aliases: dict[str, list[dict]] = defaultdict(list)
    anchors: dict[str, list[dict]] = defaultdict(list)
    for entity_id in entity_rows:
        for mention in mentions.get(entity_id, []):
            surface = mention["surface"].strip()
            # Alias eligibility is inferred only from spans visible in the
            # anonymized source; V4 maps intentionally contain no marker or
            # role metadata from the reconstruction stage.
            if _marker_like(surface) and len(surface.split()) == 1:
                aliases[entity_id].append(mention)
            elif len(surface.split()) >= 2:
                anchors[entity_id].append(mention)

    # A document may contain hundreds of repeated aliases.  Select one stable
    # representative span per identity, then consider identity pairs rather
    # than the Cartesian product of every occurrence.
    alias_choice = {
        entity_id: min(values, key=lambda x: stable_hash(doc_id, entity_id, x["start"], seed="vdt-v4-alias"))
        for entity_id, values in aliases.items()
    }
    anchor_choice = {
        entity_id: min(values, key=lambda x: stable_hash(doc_id, entity_id, x["start"], seed="vdt-v4-anchor"))
        for entity_id, values in anchors.items()
    }
    candidates = []
    for left_id, right_id in combinations(sorted(entity_rows), 2):
        for donor, recipient in ((left_id, right_id), (right_id, left_id)):
            if donor not in alias_choice or recipient not in alias_choice:
                continue
            donor_alias, recipient_alias = alias_choice[donor], alias_choice[recipient]
            donor_marker = donor_alias["surface"].strip()
            recipient_marker = recipient_alias["surface"].strip()
            if (
                donor not in anchor_choice
                or recipient not in anchor_choice
                or normalize_text(donor_marker) == normalize_text(recipient_marker)
                or len(donor_marker) != len(recipient_marker)
            ):
                continue
            if recipient_alias["start"] == donor_alias["start"]:
                continue
            candidates.append({
                "donor_id": donor,
                "recipient_id": recipient,
                "donor_alias": donor_alias,
                "recipient_alias": recipient_alias,
                "donor_anchor": anchor_choice[donor],
                "recipient_anchor": anchor_choice[recipient],
                "replacement": donor_marker,
            })
    candidates.sort(key=lambda x: stable_hash(
        doc_id, x["donor_id"], x["recipient_id"], x["recipient_alias"]["start"], seed="vdt-v4-collision"
    ))
    return candidates[:8]


def _make_collision_variant(item: dict, candidate: dict, variant_index: int, split: str,
                            context_chars: int, seed: str) -> tuple[dict, dict, list[dict]]:
    text, mapping, doc_id = item["text"], item["map"], item["doc_id"]
    target = candidate["recipient_alias"]
    replacement = candidate["replacement"]
    if len(replacement) != int(target["end"]) - int(target["start"]):
        raise ValueError("counterfactual alias must preserve the source span length")
    start, end = int(target["start"]), int(target["end"])
    variant_text = text[:start] + replacement + text[end:]
    variant_id = f"counterfactual_marker_collision_{variant_index:02d}"

    variant_mapping = {
        **mapping,
        "replacements": [dict(value) for value in mapping.get("replacements", [])],
    }
    for value in variant_mapping["replacements"]:
        if (
            str(value.get("entity_id")) == candidate["recipient_id"]
            and int(value["start"]) == start and int(value["end"]) == end
        ):
            value["original"] = replacement
            value["replacement"] = replacement
        else:
            value["replacement"] = str(value.get("original") or "")
    variant_mapping["source_sha256"] = hashlib.sha256(variant_text.encode("utf-8")).hexdigest()

    donor_alias = _mention(variant_text, candidate["donor_alias"], context_chars)
    recipient_alias_item = {**target, "original": replacement}
    recipient_alias = _mention(variant_text, recipient_alias_item, context_chars)
    donor_anchor = _mention(variant_text, candidate["donor_anchor"], context_chars)
    recipient_anchor = _mention(variant_text, candidate["recipient_anchor"], context_chars)
    metadata = _metadata(item["row"])
    variant_pairs = [
        _pair(doc_id, split, variant_text, recipient_alias, recipient_anchor, 1,
              "collision_recipient_positive", seed, variant_id, metadata),
        _pair(doc_id, split, variant_text, donor_alias, donor_anchor, 1,
              "collision_donor_positive", seed, variant_id, metadata),
        _pair(doc_id, split, variant_text, recipient_alias, donor_alias, 0,
              "collision_same_surface_negative", seed, variant_id, metadata),
    ]
    for pair in variant_pairs:
        pair["supervision"] = "constructed_from_unchanged_weak_entity_ids"
        pair["challenge_type"] = "person_marker_collision"
    variant_row = _clean_document(item["row"], variant_text, variant_id)
    variant_row["base_source_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    variant_row["counterfactual_edit"] = {
        "kind": "replace_one_short_PER_alias_with_another_existing_PER_alias",
        "start": start,
        "end": end,
        "replacement": replacement,
    }
    clean_mapping = _clean_map(variant_mapping, variant_text,
                               base_source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return variant_row, clean_mapping, variant_pairs


def _dump_jsonl(handle, rows) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _increment_pair_counts(counter: Counter, pairs: list[dict]) -> None:
    for pair in pairs:
        counter["pairs"] += 1
        counter["positive_pairs" if pair["target_linked"] else "negative_pairs"] += 1
        counter[f"difficulty:{pair['difficulty']}"] += 1


def build_dataset(
    v3_dir: Path,
    output_dir: Path,
    *,
    context_chars: int = 160,
    max_positives_per_entity: int = 8,
    max_pairs_per_document: int = 128,
    train_collision_rate: float = 0.5,
    max_train_variants_per_document: int = 3,
    seed: str = "vdt-link-v4",
) -> dict:
    """Build V4 while retaining V3's exact document splits."""
    if not 0 <= train_collision_rate <= 1:
        raise ValueError("train_collision_rate must be in [0, 1]")
    if max_train_variants_per_document < 1 or max_pairs_per_document < 2 or context_chars < 16:
        raise ValueError("variant count, pair cap, and context size are too small")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    challenge_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    natural_hard_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    location_rule_exclusions: dict[str, Counter] = {split: Counter() for split in SPLITS}
    try:
        for name in ("documents", "maps", "pairs"):
            (staging / name).mkdir()

        handles = {}
        for split in SPLITS:
            handles[("documents", split)] = (staging / "documents" / f"{split}.jsonl").open("w", encoding="utf-8")
            handles[("maps", split)] = (staging / "maps" / f"{split}.jsonl").open("w", encoding="utf-8")
            handles[("pairs", split)] = (staging / "pairs" / f"{split}.jsonl").open("w", encoding="utf-8")
            if split == "train":
                handles[("pairs", "train_natural")] = (staging / "pairs" / "train_natural.jsonl").open("w", encoding="utf-8")
            else:
                handles[("pairs", f"{split}_surface_hard")] = (staging / "pairs" / f"{split}_surface_hard.jsonl").open("w", encoding="utf-8")
                handles[("pairs", f"{split}_location_hard")] = (staging / "pairs" / f"{split}_location_hard.jsonl").open("w", encoding="utf-8")
        handles[("pairs", "train_controlled_augmented")] = (staging / "pairs" / "train_controlled_augmented.jsonl").open("w", encoding="utf-8")
        for split in ("validation", "test"):
            handles[("pairs", f"{split}_challenge")] = (staging / "pairs" / f"{split}_challenge.jsonl").open("w", encoding="utf-8")
            handles[("documents", f"{split}_challenge")] = (staging / "documents" / f"{split}_challenge.jsonl").open("w", encoding="utf-8")
            handles[("maps", f"{split}_challenge")] = (staging / "maps" / f"{split}_challenge.jsonl").open("w", encoding="utf-8")

        try:
            for split in SPLITS:
                for item in tqdm(_load_split(v3_dir, split), desc=f"Build V4 {split}", unit="doc"):
                    doc_id, text, mapping, row = item["doc_id"], item["text"], item["map"], item["row"]
                    clean_row = _clean_document(row, text)
                    clean_mapping = _clean_map(mapping, text)
                    _dump_jsonl(handles[("documents", split)], [clean_row])
                    _dump_jsonl(handles[("maps", split)], [clean_mapping])
                    counts[split]["documents"] += 1

                    metadata = _metadata(row)
                    pairs, natural_hard, excluded_pairs = build_natural_pairs(
                        doc_id, split, text, mapping, metadata,
                        context_chars=context_chars,
                        max_positives_per_entity=max_positives_per_entity,
                        max_pairs_per_document=max_pairs_per_document,
                        seed=seed,
                    )
                    location_rule_exclusions[split].update(excluded_pairs)
                    if split == "train":
                        _dump_jsonl(handles[("pairs", "train_natural")], pairs)
                    _increment_pair_counts(counts[split], pairs)
                    if split != "train":
                        _dump_jsonl(handles[("pairs", f"{split}_surface_hard")], [
                            pair for pair in natural_hard if pair["difficulty"] == "same_label_surface"
                        ])
                        _dump_jsonl(handles[("pairs", f"{split}_location_hard")], [
                            pair for pair in natural_hard if pair["difficulty"] == "location_hierarchy_conflict"
                        ])
                    natural_hard_counts[split]["same_surface_negative_pairs"] += sum(
                        pair["difficulty"] == "same_label_surface" for pair in natural_hard
                    )
                    natural_hard_counts[split]["location_hierarchy_conflict_pairs"] += sum(
                        pair["difficulty"] == "location_hierarchy_conflict" for pair in natural_hard
                    )

                    variants = _collision_candidates(item)
                    if split == "train":
                        gate = int(stable_hash(doc_id, "include_v4_collision", seed=seed)[:16], 16) / 16**16
                        variants = variants[:max_train_variants_per_document] if gate < train_collision_rate else []
                        augmented = []
                        for variant_index, candidate in enumerate(variants, 1):
                            _, _, variant_pairs = _make_collision_variant(
                                item, candidate, variant_index, split, context_chars, seed
                            )
                            augmented.extend(variant_pairs)
                        _dump_jsonl(handles[("pairs", "train_controlled_augmented")], augmented)
                        _dump_jsonl(handles[("pairs", "train")], pairs + augmented)
                        _increment_pair_counts(counts[split], augmented)
                        challenge_counts[split]["counterfactual_variants"] += len(variants)
                        challenge_counts[split]["pairs"] += len(augmented)
                    else:
                        # One held-out counterfactual per eligible document;
                        # keep it separate from natural validation/test.
                        _dump_jsonl(handles[("pairs", split)], pairs)
                        if variants:
                            variant_row, variant_mapping, variant_pairs = _make_collision_variant(
                                item, variants[0], 1, split, context_chars, seed
                            )
                            _dump_jsonl(handles[("documents", f"{split}_challenge")], [variant_row])
                            _dump_jsonl(handles[("maps", f"{split}_challenge")], [variant_mapping])
                            _dump_jsonl(handles[("pairs", f"{split}_challenge")], variant_pairs)
                            challenge_counts[split]["documents"] += 1
                            challenge_counts[split]["counterfactual_variants"] += 1
                            _increment_pair_counts(challenge_counts[split], variant_pairs)
        finally:
            for handle in handles.values():
                try:
                    handle.flush()
                except OSError:
                    pass
                finally:
                    handle.close()

        # Pair-level hard diagnostics are kept out of the primary eval sets.
        for split in SPLITS:
            if split == "train":
                continue
            hard_summary = {
                "natural_same_surface_negatives": natural_hard_counts[split]["same_surface_negative_pairs"],
                "natural_location_hierarchy_conflicts": natural_hard_counts[split]["location_hierarchy_conflict_pairs"],
            }
            challenge_counts[split].update(hard_summary)

        manifest = {
            "dataset_version": DATASET_VERSION,
            "task": "within-document linking of NER mentions in original anonymized legal text",
            "model_input": "original_anonymized_markdown; generated/reconstructed names are excluded",
            "split_policy": {
                "unit": "document",
                "policy": "reuse V3 train/validation/test files exactly; counterfactual variants remain inside their parent split",
                "source_split_overlap": 0,
            },
            "supervision": {
                "kind": "weak reconstruction-map supervision plus labels preserved under controlled edits",
                "positive": "two spans have the same V3 weak reconstruction entity_id",
                "negative": "two spans have different V3 weak reconstruction entity_id values",
                "human_annotation": False,
                "warning": "Neither natural nor counterfactual labels are independent human ground truth; V3 identity assignments can be wrong.",
            },
            "location_rule_supervision_policy": {
                "rule_source": "pinned DVHCN administrative units and Vietnamese address-unit grammar",
                "policy": "omit natural weak-map pairs that contradict a high-confidence location must-link/cannot-link; do not relabel them",
                "excluded_weak_map_pairs": {
                    split: dict(sorted(value.items()))
                    for split, value in sorted(location_rule_exclusions.items())
                },
            },
            "pair_policy": {
                "context_chars_each_side": context_chars,
                "max_positives_per_entity": max_positives_per_entity,
                "max_pairs_per_document": max_pairs_per_document,
                "negative_sampling_priority": [
                    "same-label identical surface across different weak IDs",
                    "same-label LOC pairs with conflicting province/district/commune paths",
                    "same-label token overlap",
                    "other same-label pairs; includes nearby and far pairs",
                ],
                "feature_names": list(RAW_PAIR_FEATURE_NAMES),
                "hidden_role_feature": "disabled; same_role is always zero because role is not an inference-time input",
            },
            "hard_case_policy": {
                "natural_location_diagnostics": (
                    "separate *_location_hard.jsonl for rule-detected contradictory province/district/commune paths; "
                    "canonical names and parent paths use the bundled, pinned DVHCN administrative snapshot"
                ),
                "person_collision": "replace one short marker-like PER mention with another PER entity's existing alias of equal character length; preserve all hidden IDs and all other source characters",
                "train_collision_rate": train_collision_rate,
                "max_train_variants_per_document": max_train_variants_per_document,
                "heldout_collision_variants": "one per eligible validation/test document, in separate *_challenge files; never mixed into natural main evaluation",
                "challenge_pair_composition": "two positive full-name-to-alias pairs and one different-entity same-surface negative per variant",
                "labels": "constructed from unchanged weak entity IDs; no human annotation or new identity selection",
            },
            "counts": {split: dict(sorted(value.items())) for split, value in sorted(counts.items())},
            "natural_hard_counts": {split: dict(sorted(value.items())) for split, value in sorted(natural_hard_counts.items())},
            "challenge_counts": {split: dict(sorted(value.items())) for split, value in sorted(challenge_counts.items())},
            "source": {"dataset": "outputs/entity_linking_v3_location_strict_20260924", "source_dataset_version": "entity-linking-raw-v2"},
            "privacy_sanitization": {
                "generated_name_fields_in_output": False,
                "entity_ids_in_model_pair_examples": False,
                "entity_ids_in_maps": True,
                "maps_are_training/evaluation_supervision_only": True,
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--max-positives-per-entity", type=int, default=8)
    parser.add_argument("--max-pairs-per-document", type=int, default=128)
    parser.add_argument("--train-collision-rate", type=float, default=0.5)
    parser.add_argument("--max-train-variants-per-document", type=int, default=3)
    parser.add_argument("--seed", default="vdt-link-v4")
    args = parser.parse_args()
    result = build_dataset(
        args.v3_dir.resolve(), args.output_dir.resolve(),
        context_chars=args.context_chars,
        max_positives_per_entity=args.max_positives_per_entity,
        max_pairs_per_document=args.max_pairs_per_document,
        train_collision_rate=args.train_collision_rate,
        max_train_variants_per_document=args.max_train_variants_per_document,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
