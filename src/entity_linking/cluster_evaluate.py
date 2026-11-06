"""Evaluate whole-document clusters against V4 weak reconstruction-map identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from .clustering import aggregate_cluster_metrics, cluster_metrics, complete_link_clusters
from .inference import DocumentEntityLinker
from .raw_dataset import _enrich, _target_replacements
from .training import save_json


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


def _load_scored_documents(args: argparse.Namespace, linker: DocumentEntityLinker) -> list[dict]:
    scored = []
    with args.documents.open(encoding="utf-8-sig") as documents, args.maps.open(encoding="utf-8-sig") as maps:
        records = zip_longest(documents, maps)
        for line_number, (document_line, map_line) in enumerate(
            tqdm(records, total=args.max_documents, desc="Scoring documents", unit="doc"), 1
        ):
            if args.max_documents is not None and len(scored) >= args.max_documents:
                break
            if document_line is None or map_line is None:
                raise ValueError(f"document/map line count mismatch at line {line_number}")
            if not document_line.strip() or not map_line.strip():
                raise ValueError(f"blank document/map record at line {line_number}")
            row, mapping = json.loads(document_line), json.loads(map_line)
            doc_id = str(row.get("doc_name") or row.get("case_id") or row.get("id") or "")
            if not doc_id or doc_id != str(mapping.get("doc_id") or ""):
                raise ValueError(f"document/map ID mismatch at line {line_number}")
            source = str(row.get("original_anonymized_markdown") or "")
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != mapping.get("source_sha256"):
                raise ValueError(f"document/map source hash mismatch for {doc_id}")
            text = str(row.get("entity_linking_input") or row.get("synthetic_markdown") or "")
            if not text:
                raise ValueError(f"document {doc_id} has no entity-linking input")
            entities = {str(entity["entity_id"]): entity for entity in mapping.get("entities", [])}
            translated = _target_replacements(source, text, mapping)
            mentions = []
            for item in translated:
                entity = entities.get(str(item["entity_id"]))
                if entity is None:
                    raise ValueError(f"unknown entity id in map for document {doc_id}")
                mention = _enrich(text, item, entity, linker.context_chars)
                mention["gold_entity_id"] = str(item["entity_id"])
                mentions.append(mention)
            prepared, scores, cannot_link, must_link = linker.score_document_pairs(text, mentions)
            weak_ids = [str(mention["gold_entity_id"]) for mention in prepared]
            scored.append({
                "doc_id": doc_id,
                "mentions": prepared,
                "gold_entity_ids": [str(mention["gold_entity_id"]) for mention in prepared],
                "scores": scores,
                "cannot_link": cannot_link,
                "must_link": must_link,
                "candidate_pairs_scored": len(scores),
                "location_rule_weak_label_disagreements": {
                    "linked_weak_negative": sum(weak_ids[left] != weak_ids[right]
                                                 for left, right in must_link),
                    "blocked_weak_positive": sum(weak_ids[left] == weak_ids[right]
                                                 for left, right in cannot_link),
                },
            })
    return scored


def _evaluate_threshold(scored_documents: list[dict], threshold: float) -> dict:
    per_document = []
    exact_documents = 0
    candidate_pairs = 0
    location_rule_links = 0
    location_rule_blocks = 0
    rule_label_disagreements = {"linked_weak_negative": 0, "blocked_weak_positive": 0}
    for item in scored_documents:
        clusters = complete_link_clusters(
            item["mentions"], item["scores"], threshold,
            item["cannot_link"], item.get("must_link", ()),
        )
        metrics = cluster_metrics(item["gold_entity_ids"], clusters)
        pair_counts = metrics["pairwise_counts"]
        exact_documents += int(pair_counts["fp"] == 0 and pair_counts["fn"] == 0)
        candidate_pairs += item["candidate_pairs_scored"]
        location_rule_links += len(item.get("must_link", ()))
        location_rule_blocks += len(item.get("cannot_link", ()))
        for name in rule_label_disagreements:
            rule_label_disagreements[name] += item.get("location_rule_weak_label_disagreements", {}).get(name, 0)
        per_document.append(metrics)
    result = aggregate_cluster_metrics(per_document)
    result["candidate_pairs_scored"] = candidate_pairs
    result["location_rule_links"] = location_rule_links
    result["location_rule_blocks"] = location_rule_blocks
    result["location_rule_weak_label_disagreements"] = rule_label_disagreements
    result["exact_partition_documents"] = exact_documents
    result["exact_partition_accuracy"] = exact_documents / len(scored_documents) if scored_documents else 0.0
    return result


def run(args: argparse.Namespace) -> dict:
    linker = DocumentEntityLinker(
        args.checkpoint, args.model_name, args.tokenizer, args.device,
        not args.no_fp16, args.amp_dtype, args.qlora_compute_dtype,
        args.mention_batch_size, args.pair_batch_size, args.max_length, args.context_chars,
    )
    scored_documents = _load_scored_documents(args, linker)
    if not scored_documents:
        raise ValueError("no documents found to evaluate")

    threshold = linker.threshold if args.threshold is None else args.threshold
    if args.threshold_file:
        threshold = float(json.loads(args.threshold_file.read_text(encoding="utf-8"))["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("threshold from file must be between 0 and 1")
    if args.select_threshold:
        threshold_results = [
            (candidate / 100, _evaluate_threshold(scored_documents, candidate / 100))
            for candidate in range(5, 96, 5)
        ]
        threshold, selected_metrics = max(
            threshold_results,
            key=lambda item: (
                item[1]["b_cubed"]["f1"],
                item[1]["b_cubed"]["precision"],
                item[1]["pairwise"]["precision"],
                item[0],
            ),
        )
        sweep = [
            {"threshold": candidate, "b_cubed": metrics["b_cubed"], "pairwise": metrics["pairwise"]}
            for candidate, metrics in threshold_results
        ]
    else:
        selected_metrics = _evaluate_threshold(scored_documents, threshold)
        sweep = None

    result = {
        "checkpoint": str(args.checkpoint),
        "documents": str(args.documents),
        "maps": str(args.maps),
        "threshold": threshold,
        "threshold_source": "validation cluster sweep" if args.select_threshold else (
            "validation threshold file" if args.threshold_file else (
                "explicit override" if args.threshold is not None else "checkpoint pair threshold"
            )
        ),
        "clustering": "complete_link_with_location_rules",
        "metrics": selected_metrics,
        "weak_supervision_warning": (
            "These cluster metrics compare against reconstruction-map entity IDs; they are not human gold."
        ),
    }
    if sweep is not None:
        result["threshold_sweep"] = sweep
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--documents", required=True, type=Path,
                        help="V4 documents/{validation,test}.jsonl")
    parser.add_argument("--maps", required=True, type=Path,
                        help="matching V4 maps/{validation,test}.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", default="auto")
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--threshold", type=float)
    threshold_group.add_argument("--threshold-file", type=Path,
                                 help="validation cluster metrics JSON containing the selected threshold")
    threshold_group.add_argument("--select-threshold", action="store_true",
                                 help="choose the threshold with best validation B-cubed F1; use validation data only")
    parser.add_argument("--max-documents", type=int,
                        help="optional deterministic subset for a quick smoke/threshold run")
    parser.add_argument("--mention-batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--qlora-compute-dtype", choices=["fp16", "bf16"])
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("max documents must be positive")
    for field in ("checkpoint", "documents", "maps"):
        path = getattr(args, field).resolve()
        if not path.is_file():
            parser.error(f"{field} does not exist: {path}")
        setattr(args, field, path)
    args.output = args.output.resolve()
    if args.tokenizer:
        args.tokenizer = args.tokenizer.resolve()
    if args.threshold_file:
        args.threshold_file = args.threshold_file.resolve()
        if not args.threshold_file.is_file():
            parser.error(f"threshold file does not exist: {args.threshold_file}")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
