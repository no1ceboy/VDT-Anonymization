"""Build a deterministic challenge set from rejected reconstructions.

Challenge records are deliberately not training-clean. They preserve the
synthetic reconstruction and attach the failure signals that make each record
useful for evaluation and future pipeline work.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict


MARKER_PATTERNS = {
    marker: re.compile(rf"(?<!\w){marker}(?!\w)")
    for marker in ("H1", "H2", "B1", "B2")
}

REASON_CATEGORIES = (
    "ambiguous_identity",
    "unsupported_name_prefix",
    "organization_without_form",
    "unconfirmed_marker",
    "missing_identity_anchor",
    "conflicting_location_parents",
    "location_without_address_structure",
    "fragmented_initial",
    "invalid_location_hierarchy",
)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def row_id(row, line_number):
    return str(row.get("doc_id", row.get("case_id", row.get("id", line_number))))


def quality_for(row, audit):
    return row.get("reconstruction_quality") or audit.get("reconstruction_quality") or {}


def is_clean(quality):
    if "selected" in quality:
        return bool(quality["selected"])
    return bool(
        quality.get("eligible")
        and quality.get("quality_score", 0) >= 85
        and not quality.get("reasons")
    )


def challenge_categories(row, audit):
    """Return deterministic marker/reason signals for one rejected record."""
    quality = quality_for(row, audit)
    if is_clean(quality):
        return []

    categories = []
    text = str(row.get("original_anonymized_markdown", row.get("markdown", "")))
    for marker, pattern in MARKER_PATTERNS.items():
        if pattern.search(text):
            categories.append(f"marker_{marker}")

    reasons = set(quality.get("reasons") or audit.get("review_reasons") or [])
    categories.extend(reason for reason in REASON_CATEGORIES if reason in reasons)
    return categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-file", required=True)
    parser.add_argument("--links-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--max-per-category", type=int, default=5,
                        help="Maximum selected documents per marker/reason category")
    args = parser.parse_args()
    if args.max_per_category < 1:
        parser.error("--max-per-category must be positive")
    for path in (args.synthetic_file, args.links_file):
        if not os.path.exists(path):
            parser.error(f"Input file does not exist: {path}")

    synthetic = list(iter_jsonl(args.synthetic_file))
    audits = list(iter_jsonl(args.links_file))
    if len(synthetic) != len(audits):
        raise ValueError(f"Input row mismatch: synthetic={len(synthetic)}, links={len(audits)}")

    candidates = []
    category_counts = Counter()
    for (synthetic_line, row), (audit_line, audit) in zip(synthetic, audits):
        document_id = row_id(row, synthetic_line)
        audit_id = row_id(audit, audit_line)
        if document_id != audit_id:
            raise ValueError(f"Input ID mismatch at lines {synthetic_line}/{audit_line}: {document_id!r} != {audit_id!r}")
        categories = challenge_categories(row, audit)
        for category in categories:
            category_counts[category] += 1
        if categories:
            candidates.append((document_id, row, audit, categories))

    selected_by_category = defaultdict(list)
    for document_id, row, audit, categories in sorted(candidates, key=lambda item: item[0]):
        for category in categories:
            if len(selected_by_category[category]) < args.max_per_category:
                selected_by_category[category].append(document_id)

    selected_ids = set().union(*selected_by_category.values()) if selected_by_category else set()
    selected_rows = []
    for document_id, row, audit, categories in candidates:
        if document_id not in selected_ids:
            continue
        output_row = dict(row)
        quality = quality_for(row, audit)
        output_row["challenge"] = {
            "categories": categories,
            "marker_hits": {
                marker: len(MARKER_PATTERNS[marker].findall(
                    str(row.get("original_anonymized_markdown", row.get("markdown", "")))
                ))
                for marker in MARKER_PATTERNS
            },
            "quality_score": quality.get("quality_score"),
            "quality_tier": quality.get("quality_tier"),
            "reasons": quality.get("reasons") or audit.get("review_reasons") or [],
            "source_audit_doc_id": document_id,
        }
        selected_rows.append(output_row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "challenge_version": "challenge-v1",
        "policy": {
            "max_per_category": args.max_per_category,
            "clean_records_excluded": True,
            "selection": "rejected records with marker-family or representative quality signals",
        },
        "counts": {
            "input_documents": len(synthetic),
            "candidate_documents": len(candidates),
            "selected_documents": len(selected_rows),
        },
        "available_category_counts": dict(sorted(category_counts.items())),
        "selected_category_counts": {
            category: len(documents)
            for category, documents in sorted(selected_by_category.items())
        },
        "selected_documents": [row_id(row, index) for index, row in enumerate(selected_rows)],
    }
    with open(args.report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[CHALLENGE] Input documents: {len(synthetic)}")
    print(f"[CHALLENGE] Selected documents: {len(selected_rows)}")
    print(f"[CHALLENGE] Dataset: {args.output_file}")
    print(f"[CHALLENGE] Report: {args.report_file}")


if __name__ == "__main__":
    main()
