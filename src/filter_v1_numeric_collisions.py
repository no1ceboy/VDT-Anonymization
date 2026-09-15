"""Conservatively salvage a legacy clean JSONL after numeric-boundary bugs.

Rows and maps are copied unchanged only when they pass the post-filter.  This
script never attempts to repair an already generated target: ambiguous rows
are removed and should be refilled by the current production pipeline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path

try:
    from .reconstruction import IDENTIFIER_PREFIX, numeric_suffix
except ImportError:
    from reconstruction import IDENTIFIER_PREFIX, numeric_suffix


FILTER_VERSION = "legacy-numeric-postfilter-v1"
AMOUNT_CONTINUATION_RE = re.compile(r"^[.,]\d{3}(?:[.,]\d{3})*(?:\s*(?:đ|đồng|vnđ))?", re.I)


def row_id(row: dict) -> str:
    for field in ("doc_name", "case_id", "official_document_id", "id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def future_issued_date(row: dict, today: dt.date | None = None) -> bool:
    value = str(row.get("issued_date") or "").strip()
    if not value:
        return False
    try:
        issued = dt.date.fromisoformat(value[:10])
    except ValueError:
        return False
    return issued > (today or dt.date.today())


def semantic_rejection_reasons(row: dict, mapping: dict) -> list[str]:
    source = str(row.get("original_anonymized_markdown") or "")
    reasons = set()
    entities = mapping.get("entities") or []
    replacements = mapping.get("replacements") or []
    replacement_counts = Counter(item.get("entity_id") for item in replacements)

    for replacement in replacements:
        start = replacement.get("start")
        end = replacement.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(source)):
            reasons.add("invalid_replacement_span")
            continue
        if source[start:end] != replacement.get("original"):
            reasons.add("source_span_mismatch")
            continue
        original = str(replacement.get("original") or "")
        if numeric_suffix(original) is not None and AMOUNT_CONTINUATION_RE.match(source[end:end + 40]):
            reasons.add("numeric_amount_boundary_collision")
        before = source[max(0, start - 150):start]
        if numeric_suffix(original) is not None and IDENTIFIER_PREFIX.search(before):
            reasons.add("document_identifier_collision")
        if numeric_suffix(original) is not None and re.match(r"\s*/\s*\d", source[end:end + 20]):
            reasons.add("compound_literal_number_collision")

    for entity in entities:
        suffix = numeric_suffix(entity.get("marker"))
        if suffix is None or suffix < 20:
            continue
        if entity.get("label") != "PER" or replacement_counts[entity.get("entity_id")] < 2:
            reasons.add("unsupported_large_numeric_suffix")

    return sorted(reasons)


def write_atomic_pair(clean_input: Path, maps_input: Path, clean_output: Path,
                      maps_output: Path, report_output: Path) -> dict:
    for path in (clean_output, maps_output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    clean_temp = clean_output.with_suffix(clean_output.suffix + ".tmp")
    maps_temp = maps_output.with_suffix(maps_output.suffix + ".tmp")
    seen_hashes = set()
    counts = Counter()
    rejection_counts = Counter()
    categories = Counter()

    with (
        clean_input.open("r", encoding="utf-8-sig") as clean_handle,
        maps_input.open("r", encoding="utf-8-sig") as maps_handle,
        clean_temp.open("w", encoding="utf-8") as clean_out,
        maps_temp.open("w", encoding="utf-8") as maps_out,
    ):
        for line_number, pair in enumerate(zip_longest(clean_handle, maps_handle), 1):
            clean_line, map_line = pair
            counts["input_rows"] += 1
            reasons = []
            if clean_line is None or map_line is None:
                reasons.append("input_line_count_mismatch")
            else:
                try:
                    row = json.loads(clean_line)
                    mapping = json.loads(map_line)
                except json.JSONDecodeError:
                    reasons.append("invalid_json")
                else:
                    doc_id = row_id(row)
                    if not doc_id or str(mapping.get("doc_id")) != doc_id:
                        reasons.append("map_id_mismatch")
                    source = str(row.get("original_anonymized_markdown") or "")
                    source_hash = hashlib.sha256(source.encode()).hexdigest()
                    if not source or source_hash != mapping.get("source_sha256"):
                        reasons.append("source_hash_mismatch")
                    if source_hash in seen_hashes:
                        reasons.append("duplicate_source_text")
                    if future_issued_date(row):
                        reasons.append("future_issued_date")
                    quality = row.get("reconstruction_quality") or {}
                    if not (
                        quality.get("eligible") is True
                        and quality.get("quality_score") == 100
                        and not quality.get("reasons")
                        and quality.get("roundtrip_verified") is True
                        and int(quality.get("replacements", 0)) >= 1
                    ):
                        reasons.append("legacy_quality_gate_failed")
                    reasons.extend(semantic_rejection_reasons(row, mapping))

            reasons = sorted(set(reasons))
            if reasons:
                counts["rejected_rows"] += 1
                rejection_counts.update(reasons)
                continue

            seen_hashes.add(source_hash)
            clean_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            maps_out.write(json.dumps(mapping, ensure_ascii=False) + "\n")
            counts["retained_rows"] += 1
            categories[str(row.get("category") or "Unknown")] += 1

        for handle in (clean_out, maps_out):
            handle.flush()
            os.fsync(handle.fileno())

    os.replace(clean_temp, clean_output)
    os.replace(maps_temp, maps_output)
    report = {
        "filter_version": FILTER_VERSION,
        "policy": "reject_not_repair",
        "counts": dict(counts),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "retained_categories": dict(sorted(categories.items())),
        "input_clean": str(clean_input),
        "input_maps": str(maps_input),
        "output_clean": str(clean_output),
        "output_maps": str(maps_output),
    }
    report_temp = report_output.with_suffix(report_output.suffix + ".tmp")
    with report_temp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_temp, report_output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-input", required=True)
    parser.add_argument("--maps-input", required=True)
    parser.add_argument("--clean-output", required=True)
    parser.add_argument("--maps-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()
    inputs = [Path(args.clean_input).resolve(), Path(args.maps_input).resolve()]
    outputs = [Path(args.clean_output).resolve(), Path(args.maps_output).resolve(), Path(args.report_output).resolve()]
    if any(not path.is_file() for path in inputs):
        parser.error("both input files must exist")
    if len(set(outputs)) != 3 or any(path in inputs for path in outputs):
        parser.error("outputs must be distinct and cannot overwrite inputs")
    report = write_atomic_pair(*inputs, *outputs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
