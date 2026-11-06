"""Reproduce the person-name baseline described in the VDT phase 2 report.

The report sorts PER mentions by document offset and maps a later mention to
an earlier mention when the later surface is a literal substring of the
earlier surface. LOC handling remains the project's existing rule-based path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations, zip_longest
from pathlib import Path

from .baseline import metrics
from .clustering import aggregate_cluster_metrics, cluster_metrics, complete_link_clusters
from .dataset import normalize_text
from .location_constraints import location_rule_decision
from .raw_dataset import _enrich, _target_replacements


BASELINE_NAME = "vdt_report_algorithm_1_ordered_substring_plus_existing_location_rules"


def report_person_roots(mentions: list[dict]) -> dict[int, int]:
    """Return mention-index -> report mapping root for PER mentions.

    This follows Algorithm 1's ordered, case-sensitive substring check. NFC
    normalization only makes canonically equivalent Unicode sequences equal;
    it does not fold case or normalize whitespace.
    """
    person_indices = [
        index for index, mention in enumerate(mentions)
        if str(mention.get("label") or "").upper() == "PER"
    ]
    surfaces = [
        unicodedata.normalize("NFC", str(mentions[index].get("surface") or ""))
        for index in person_indices
    ]
    marks = [-1] * len(person_indices)
    for left in range(len(person_indices)):
        for right in range(left, len(person_indices)):
            if marks[right] == -1 and surfaces[right] in surfaces[left]:
                marks[right] = left
    return {
        mention_index: marks[index]
        for index, mention_index in enumerate(person_indices)
    }


def _pair_metrics(rows: list[dict], owner_by_doc: dict[str, dict[tuple[int, int], int]]) -> dict:
    total = Counter()
    by_difficulty: dict[str, Counter] = defaultdict(Counter)
    by_challenge: dict[str, Counter] = defaultdict(Counter)
    by_category: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    missed = 0
    for row in rows:
        doc_id = str(row["doc_id"])
        owner = owner_by_doc.get(doc_id, {})
        a, b = row["mention_a"], row["mention_b"]
        left = (int(a["start"]), int(a["end"]))
        right = (int(b["start"]), int(b["end"]))
        if left not in owner or right not in owner:
            missed += 1
            raise ValueError(f"Pair {row.get('pair_id')} does not align to the document/map mention spans")
        prediction = int(owner[left] == owner[right])
        target = int(row["target_linked"])
        outcome = "tp" if target and prediction else "fn" if target else "fp" if prediction else "tn"
        total[outcome] += 1
        difficulty = str(row.get("difficulty") or "Unknown")
        by_difficulty[difficulty][outcome] += 1
        metadata = row.get("document_metadata") or {}
        by_challenge[str(metadata.get("primary_challenge") or "Unknown")][outcome] += 1
        by_category[str(metadata.get("category") or "Unknown")][outcome] += 1
        if len(examples[difficulty]) < 3:
            examples[difficulty].append({
                "pair_id": row.get("pair_id"),
                "doc_id": doc_id,
                "surface_a": a.get("surface"),
                "surface_b": b.get("surface"),
                "start_a": left[0],
                "end_a": left[1],
                "start_b": right[0],
                "end_b": right[1],
                "target_linked": target,
                "predicted_linked": prediction,
            })
    return {
        "evaluation_kind": "agreement with weak reconstruction labels; not independent human-ground-truth accuracy",
        "baseline": BASELINE_NAME,
        "rule": (
            "For PER, sort mentions by start offset and follow VDT phase 2 Algorithm 1: "
            "a later surface maps to an earlier surface when it is a literal substring "
            "(case-sensitive; NFC-normalized). For LOC, retain the existing canonical "
            "location rules and normalized-surface fallback. Other labels use normalized-surface matching."
        ),
        "examples_by_difficulty": dict(examples),
        "metrics": {
            "overall": metrics(total),
            "by_difficulty": {name: metrics(counter) for name, counter in sorted(by_difficulty.items())},
            "by_primary_challenge": {name: metrics(counter) for name, counter in sorted(by_challenge.items())},
            "by_document_category": {name: metrics(counter) for name, counter in sorted(by_category.items())},
        },
        "pair_span_alignment_misses": missed,
    }


def evaluate(documents_path: Path, maps_path: Path, pairs_path: Path) -> tuple[dict, dict, dict[str, dict]]:
    owner_by_doc: dict[str, dict[tuple[int, int], int]] = {}
    per_document_metrics = []
    metrics_by_document_type: dict[str, list[dict]] = defaultdict(list)
    exact_documents = 0
    location_counts = Counter()
    with pairs_path.open(encoding="utf-8-sig") as handle:
        pair_rows = [json.loads(line) for line in handle if line.strip()]
    with documents_path.open(encoding="utf-8-sig") as documents, maps_path.open(encoding="utf-8-sig") as maps:
        for line_number, (document_line, map_line) in enumerate(zip_longest(documents, maps), 1):
            if document_line is None or map_line is None:
                raise ValueError(f"document/map count mismatch at line {line_number}")
            if not document_line.strip() or not map_line.strip():
                raise ValueError(f"blank document/map record at line {line_number}")
            row, mapping = json.loads(document_line), json.loads(map_line)
            doc_id = str(row.get("doc_name") or row.get("case_id") or row.get("id") or "")
            if not doc_id or doc_id != str(mapping.get("doc_id") or ""):
                raise ValueError(f"document/map ID mismatch at line {line_number}")
            source = str(row.get("original_anonymized_markdown") or "")
            source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source_hash != mapping.get("source_sha256"):
                raise ValueError(f"document/map source hash mismatch for {doc_id}")
            text = str(row.get("entity_linking_input") or row.get("synthetic_markdown") or "")
            if not text:
                raise ValueError(f"document {doc_id} has no entity-linking input")
            translated = _target_replacements(source, text, mapping)
            entities = {str(entity["entity_id"]): entity for entity in mapping.get("entities", [])}
            mentions = []
            for item in translated:
                entity = entities.get(str(item["entity_id"]))
                if entity is None:
                    raise ValueError(f"unknown entity ID in document {doc_id}")
                mention = _enrich(text, item, entity, context_chars=160)
                mention["gold_entity_id"] = str(item["entity_id"])
                mentions.append(mention)
            mentions.sort(key=lambda mention: (int(mention["start"]), int(mention["end"])))

            person_roots = report_person_roots(mentions)
            scores: dict[tuple[int, int], float] = {}
            must_link = []
            cannot_link = []
            for left, right in combinations(range(len(mentions)), 2):
                mention_a, mention_b = mentions[left], mentions[right]
                if str(mention_a.get("label") or "").casefold() != str(mention_b.get("label") or "").casefold():
                    continue
                label = str(mention_a.get("label") or "").upper()
                if label == "PER":
                    if person_roots[left] == person_roots[right]:
                        scores[(left, right)] = 1.0
                    continue
                if label == "LOC":
                    decision = location_rule_decision({"mention_a": mention_a, "mention_b": mention_b})
                    if decision == "link":
                        must_link.append((left, right))
                        location_counts["links"] += 1
                        if mention_a["gold_entity_id"] != mention_b["gold_entity_id"]:
                            location_counts["weak_label_disagreements"] += 1
                        continue
                    if decision == "block":
                        cannot_link.append((left, right))
                        location_counts["blocks"] += 1
                        if mention_a["gold_entity_id"] == mention_b["gold_entity_id"]:
                            location_counts["weak_label_disagreements"] += 1
                        continue
                if normalize_text(mention_a.get("surface")) == normalize_text(mention_b.get("surface")):
                    scores[(left, right)] = 1.0

            clusters = complete_link_clusters(mentions, scores, 0.5, cannot_link, must_link)
            gold_ids = [mention["gold_entity_id"] for mention in mentions]
            cluster_result = cluster_metrics(gold_ids, clusters)
            per_document_metrics.append(cluster_result)
            challenge_type = str(row.get("entity_linking_challenge_document_type") or "Unknown")
            metrics_by_document_type[challenge_type].append(cluster_result)
            counts = cluster_result["pairwise_counts"]
            exact_documents += int(counts["fp"] == 0 and counts["fn"] == 0)

            owner = {}
            for cluster_index, cluster in enumerate(clusters):
                for mention_index in cluster:
                    mention = mentions[mention_index]
                    owner[(int(mention["start"]), int(mention["end"]))] = cluster_index
            owner_by_doc[doc_id] = owner

    pair_result = _pair_metrics(pair_rows, owner_by_doc)
    document_result = {
        "evaluation_kind": "agreement with weak reconstruction-map entity IDs; not human gold",
        "baseline": BASELINE_NAME,
        "inputs": {
            "documents": str(documents_path),
            "maps": str(maps_path),
            "pairs": str(pairs_path),
        },
        "documents": aggregate_cluster_metrics(per_document_metrics),
        "exact_partition_documents": exact_documents,
        "exact_partition_accuracy": exact_documents / len(per_document_metrics) if per_document_metrics else 0.0,
        "by_challenge_document_type": {
            name: aggregate_cluster_metrics(values)
            for name, values in sorted(metrics_by_document_type.items())
        },
        "location_rule_counts": {
            "links": location_counts["links"],
            "blocks": location_counts["blocks"],
            "weak_label_disagreements": location_counts["weak_label_disagreements"],
        },
    }
    return pair_result, document_result, owner_by_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--pair-output", required=True, type=Path)
    parser.add_argument("--document-output", required=True, type=Path)
    args = parser.parse_args()
    pair_result, document_result, _ = evaluate(args.documents, args.maps, args.pairs)
    for path, result in ((args.pair_output, pair_result), (args.document_output, document_result)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"pairwise": pair_result["metrics"]["overall"], "documents": document_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
