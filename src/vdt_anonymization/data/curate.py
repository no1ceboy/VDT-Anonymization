"""Select a diverse, marker-bearing source subset before NER inference.

Selection is deterministic: each document is ranked by a stable hash within a
legal stratum.  The source text and metadata are preserved unchanged.
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict


PROCEDURAL_RE = re.compile(r"(?<!\w)(?:NLQ|NLC)\s*\d+(?!\w)", re.I)
DOTTED_RE = re.compile(r"(?<!\w)(?:[A-ZĐ]\.){2,5}[A-ZĐ](?:[1-9]\d*)?(?!\w)")
PERSON_MARKER_RE = re.compile(
    r"(?<!\w)(?:ông|bà|anh|chị|cháu|cô|chú|bác|em|bị cáo|nguyên đơn|"
    r"bị đơn|bị hại|người làm chứng|Ã´ng|bÃ |anh|chá»‹|chÃ¡u|cÃ´|"
    r"chÃº|bÃ¡c|bá»‹ cÃ¡o|nguyÃªn Ä‘Æ¡n|bá»‹ Ä‘Æ¡n|bá»‹ háº¡i|"
    r"ngÆ°á»i lÃ m chá»©ng)\s+[A-ZĐ](?:[ \t]*[1-9]\d*)?(?!\w)",
    re.I,
)
ADDRESS_ALIAS_RE = re.compile(
    r"(?<!\w)(?:số(?: nhà)?|tầng|lầu|phòng|căn|thửa|lô|ấp|thôn|làng|khóm|"
    r"khu phố|tổ(?: dân phố)?|xã|phường|thị trấn|huyện|quận|thị xã|"
    r"thành phố|tỉnh|đường|lộ|quốc lộ|ngõ|hẻm)\s+[A-ZĐ](?:[ \t]*[1-9]\d*)?(?!\w)",
    re.I,
)


def interest_features(text):
    """Return counts for patterns relevant to reconstruction."""
    text = str(text or "")
    procedural = PROCEDURAL_RE.findall(text)
    dotted = DOTTED_RE.findall(text)
    address = ADDRESS_ALIAS_RE.findall(text)
    numbered = PERSON_MARKER_RE.findall(text)
    feature_counts = {
        "procedural_codes": len(procedural),
        "dotted_initials": len(dotted),
        "address_aliases": len(address),
        "person_aliases": len(numbered),
    }
    feature_counts["interest_score"] = (
        4 * feature_counts["procedural_codes"]
        + 4 * feature_counts["dotted_initials"]
        + 3 * feature_counts["address_aliases"]
        + min(feature_counts["person_aliases"], 8)
    )
    return feature_counts


def stratum(row):
    values = []
    for field in ("case_type", "doc_type", "cap_xet_xu"):
        value = row.get(field)
        values.append(str(value).strip() if value not in (None, "") else "unknown")
    return "|".join(values)


def stable_rank(seed, doc_id):
    digest = hashlib.sha256(f"{seed}|{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--manifest-file")
    parser.add_argument("--per-stratum", type=int, default=100,
                        help="Maximum selected documents per case/doc/proceeding stratum")
    parser.add_argument("--max-documents", type=int, default=0,
                        help="Optional global cap after balanced stratum selection")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--include-no-marker", action="store_true",
                        help="Include documents without a targeted marker")
    parser.add_argument("--seed", default="vdt-curation-v1")
    args = parser.parse_args()
    if args.per_stratum < 1 or args.max_documents < 0 or args.min_chars < 0:
        parser.error("per-stratum must be positive; max-documents and min-chars cannot be negative")
    if not os.path.exists(args.input_file):
        parser.error(f"Input file does not exist: {args.input_file}")
    if os.path.normcase(os.path.realpath(args.input_file)) == os.path.normcase(os.path.realpath(args.output_file)):
        parser.error("Output cannot overwrite input")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    manifest_file = args.manifest_file or os.path.splitext(args.output_file)[0] + ".manifest.json"
    os.makedirs(os.path.dirname(os.path.abspath(manifest_file)), exist_ok=True)

    selected = defaultdict(list)
    totals = Counter()
    feature_totals = Counter()
    for line_number, row in iter_jsonl(args.input_file):
        totals["seen"] += 1
        text = row.get("markdown")
        if not isinstance(text, str) or len(text.strip()) < args.min_chars:
            totals["too_short"] += 1
            continue
        features = interest_features(text)
        feature_totals.update({k: v for k, v in features.items() if k != "interest_score"})
        if not args.include_no_marker and features["interest_score"] <= 0:
            totals["without_interest_marker"] += 1
            continue
        totals["eligible"] += 1
        doc_id = str(row.get("case_id", row.get("doc_name", row.get("id", line_number))))
        item = (stable_rank(args.seed, doc_id), line_number, row, features)
        bucket = selected[stratum(row)]
        bucket.append(item)
        bucket.sort(key=lambda value: value[0])
        if len(bucket) > args.per_stratum:
            bucket.pop()

    items = [item for bucket in selected.values() for item in bucket]
    if args.max_documents:
        items.sort(key=lambda value: (value[0], value[1]))
        items = items[:args.max_documents]
    items.sort(key=lambda value: value[1])
    with open(args.output_file, "w", encoding="utf-8") as handle:
        for rank, line_number, row, features in items:
            output = dict(row)
            output["curation"] = {
                "curation_version": "curation-v1",
                "source_row_number": line_number,
                "stratum": stratum(row),
                "stable_seed": args.seed,
                "interest_features": features,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    totals["selected"] = len(items)
    manifest = {
        "curation_version": "curation-v1",
        "policy": {
            "per_stratum": args.per_stratum,
            "max_documents": args.max_documents,
            "min_chars": args.min_chars,
            "include_no_marker": args.include_no_marker,
            "seed": args.seed,
            "stratum_fields": ["case_type", "doc_type", "cap_xet_xu"],
        },
        "counts": dict(totals),
        "strata": len(selected),
        "feature_totals": dict(feature_totals),
        "output_file": args.output_file,
    }
    with open(manifest_file, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[CURATE] Seen: {totals['seen']}; eligible: {totals['eligible']}; selected: {totals['selected']}")
    print(f"[CURATE] Strata: {len(selected)}")
    print(f"[CURATE] Dataset: {args.output_file}")
    print(f"[CURATE] Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
