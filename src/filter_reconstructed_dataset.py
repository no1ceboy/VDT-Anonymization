"""Filter reconstruction output into conservative training data.

This command streams the synthetic dataset and link audit in lockstep, so it
does not load a large corpus or its full audit into memory.
"""

import argparse
import json
import os
from collections import Counter

try:
    from .reconstruction_quality import QUALITY_VERSION, score_document
except ImportError:
    from reconstruction_quality import QUALITY_VERSION, score_document


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
    for path in (args.synthetic_file, args.links_file):
        if not os.path.exists(path):
            parser.error(f"Input file does not exist: {path}")
    if os.path.normcase(os.path.realpath(args.clean_output)) in {
        os.path.normcase(os.path.realpath(args.synthetic_file)),
        os.path.normcase(os.path.realpath(args.links_file)),
    }:
        parser.error("Output cannot overwrite an input")
    if os.path.normcase(os.path.realpath(args.report_file)) in {
        os.path.normcase(os.path.realpath(args.synthetic_file)),
        os.path.normcase(os.path.realpath(args.links_file)),
    }:
        parser.error("Report cannot overwrite an input")
    os.makedirs(os.path.dirname(os.path.abspath(args.clean_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)

    totals = Counter()
    score_buckets = Counter()
    reason_counts = Counter()
    by_case_type = Counter()
    by_doc_type = Counter()
    selected_by_case_type = Counter()
    selected_by_doc_type = Counter()

    synthetic_rows = iter_jsonl(args.synthetic_file)
    link_rows = iter_jsonl(args.links_file)
    with open(args.clean_output, "w", encoding="utf-8") as clean_handle:
        for (synthetic_line, row), (link_line, audit) in zip(synthetic_rows, link_rows):
            totals["documents"] += 1
            synthetic_id = doc_id(row, synthetic_line)
            audit_id = doc_id(audit, link_line)
            if synthetic_id != audit_id:
                raise ValueError(
                    f"Input order/id mismatch at lines {synthetic_line}/{link_line}: "
                    f"{synthetic_id!r} != {audit_id!r}"
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
                totals["selected"] += 1
                selected_by_case_type[case_type] += 1
                selected_by_doc_type[doc_type] += 1
            else:
                totals["rejected"] += 1

    try:
        next(synthetic_rows)
        raise ValueError("Synthetic dataset has more rows than link audit")
    except StopIteration:
        pass
    try:
        next(link_rows)
        raise ValueError("Link audit has more rows than synthetic dataset")
    except StopIteration:
        pass

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

