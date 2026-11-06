"""Filter reconstruction output into conservative training data.

This command streams the synthetic dataset and link audit in lockstep, so it
does not load a large corpus or its full audit into memory.
"""

import argparse
import json
import os
from collections import Counter
from itertools import zip_longest

from ..core.quality import QUALITY_VERSION, score_document


DEFAULT_SYNTHETIC = "outputs/synthetic_unanonymized.jsonl"
DEFAULT_LINKS = "outputs/entity_links.jsonl"
DEFAULT_CLEAN = "outputs/synthetic_unanonymized_clean.jsonl"
DEFAULT_REPORT = "outputs/reconstruction_quality.json"


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def doc_id(row, line_number):
    return str(row.get("doc_id", row.get("case_id", row.get("id", line_number))))


def metadata_bucket(row, field):
    value = row.get(field)
    return str(value).strip() if value not in (None, "") else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-file", default=DEFAULT_SYNTHETIC)
    parser.add_argument("--links-file", default=DEFAULT_LINKS)
    parser.add_argument("--clean-output", default=DEFAULT_CLEAN)
    parser.add_argument("--report-file", default=DEFAULT_REPORT)
    parser.add_argument("--maps-input", help="Optional reconstruction map JSONL aligned with the synthetic input")
    parser.add_argument("--maps-output", help="Write maps for exactly the selected clean rows")
    parser.add_argument("--min-score", type=int, default=85,
                        help="Minimum score for output; default 85")
    parser.add_argument("--min-replacements", type=int, default=1,
                        help="Require at least this many applied replacements")
    parser.add_argument("--allow-review", action="store_true",
                        help="Allow documents with review reasons when score passes")
    args = parser.parse_args()

    if not 0 <= args.min_score <= 100:
        parser.error("--min-score must be between 0 and 100")
    if args.min_replacements < 0:
        parser.error("--min-replacements cannot be negative")
    if bool(args.maps_input) != bool(args.maps_output):
        parser.error("--maps-input and --maps-output must be supplied together")
    input_paths = [args.synthetic_file, args.links_file]
    output_paths = [args.clean_output, args.report_file]
    if args.maps_input:
        input_paths.append(args.maps_input)
        output_paths.append(args.maps_output)
    for path in input_paths:
        if not os.path.exists(path):
            parser.error(f"Input file does not exist: {path}")
    normalized_inputs = {os.path.normcase(os.path.realpath(path)) for path in input_paths}
    normalized_outputs = [os.path.normcase(os.path.realpath(path)) for path in output_paths]
    if len(set(normalized_outputs)) != len(normalized_outputs) or any(
        path in normalized_inputs for path in normalized_outputs
    ):
        parser.error("Outputs must be distinct and must not overwrite an input")
    os.makedirs(os.path.dirname(os.path.abspath(args.clean_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
    if args.maps_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.maps_output)), exist_ok=True)

    totals = Counter()
    score_buckets = Counter()
    reason_counts = Counter()
    by_case_type = Counter()
    by_doc_type = Counter()
    selected_by_case_type = Counter()
    selected_by_doc_type = Counter()

    synthetic_rows = iter_jsonl(args.synthetic_file)
    link_rows = iter_jsonl(args.links_file)
    maps_rows = iter_jsonl(args.maps_input) if args.maps_input else None
    with open(args.clean_output, "w", encoding="utf-8") as clean_handle:
        maps_handle = open(args.maps_output, "w", encoding="utf-8") if args.maps_output else None
        try:
            rows = zip_longest(synthetic_rows, link_rows, maps_rows) if maps_rows else zip_longest(
                synthetic_rows, link_rows
            )
            for values in rows:
                if maps_rows:
                    synthetic_item, link_item, maps_item = values
                else:
                    synthetic_item, link_item = values
                    maps_item = None
                if synthetic_item is None or link_item is None or (maps_rows and maps_item is None):
                    raise ValueError("Synthetic, link-audit, and map inputs must have matching row counts")
                synthetic_line, row = synthetic_item
                link_line, audit = link_item
                mapping = maps_item[1] if maps_item else None
                totals["documents"] += 1
                synthetic_id = doc_id(row, synthetic_line)
                audit_id = doc_id(audit, link_line)
                if synthetic_id != audit_id:
                    raise ValueError(
                        f"Input order/id mismatch at lines {synthetic_line}/{link_line}: "
                        f"{synthetic_id!r} != {audit_id!r}"
                    )
                if mapping is not None and str(mapping.get("doc_id")) != synthetic_id:
                    raise ValueError(
                        f"Reconstruction map mismatch at line {maps_item[0]}: "
                        f"{mapping.get('doc_id')!r} != {synthetic_id!r}"
                    )
                quality = score_document(audit, row, args.min_replacements)
                quality["selected"] = bool(
                    quality["quality_score"] >= args.min_score
                    and (args.allow_review or not quality["reasons"])
                )
                row["reconstruction_quality"] = quality
                score_buckets[quality["quality_tier"]] += 1
                reason_counts.update(quality["reason_counts"])
                case_type = metadata_bucket(row, "case_type")
                doc_type = metadata_bucket(row, "doc_type")
                by_case_type[case_type] += 1
                by_doc_type[doc_type] += 1
                if quality["selected"]:
                    clean_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if maps_handle:
                        maps_handle.write(json.dumps(mapping, ensure_ascii=False) + "\n")
                    totals["selected"] += 1
                    selected_by_case_type[case_type] += 1
                    selected_by_doc_type[doc_type] += 1
                else:
                    totals["rejected"] += 1
        finally:
            if maps_handle:
                maps_handle.close()

    report = {
        "quality_version": QUALITY_VERSION,
        "policy": {
            "min_score": args.min_score,
            "min_replacements": args.min_replacements,
            "allow_review": args.allow_review,
            "selection_rule": "score >= min_score and no review reasons unless --allow-review",
        },
        "counts": dict(totals),
        "score_tiers": dict(score_buckets),
        "reason_counts": dict(reason_counts.most_common()),
        "input_distribution": {
            "case_type": dict(by_case_type.most_common()),
            "doc_type": dict(by_doc_type.most_common()),
        },
        "selected_distribution": {
            "case_type": dict(selected_by_case_type.most_common()),
            "doc_type": dict(selected_by_doc_type.most_common()),
        },
    }
    with open(args.report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[FILTER] Documents: {totals['documents']}; selected: {totals['selected']}; rejected: {totals['rejected']}")
    print(f"[FILTER] Clean dataset: {args.clean_output}")
    print(f"[FILTER] Report: {args.report_file}")


if __name__ == "__main__":
    main()
