"""Build a browsable HTML review folder for a JSONL demo run."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from urllib.parse import quote

try:
    from .inspect_entity_links import build_html, document_id
except ImportError:
    from inspect_entity_links import build_html, document_id


DEFAULT_SOURCE = "datasets/demo_court_documents_v2/documents.jsonl"
DEFAULT_LINKS = "outputs/entity_links.jsonl"
DEFAULT_SYNTHETIC = "outputs/synthetic_unanonymized.jsonl"
DEFAULT_OUTPUT = "outputs/demo_500_html"


def load_jsonl(path, id_field=None):
    records = {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get(id_field)) if id_field else document_id(row, row_number)
            if key in records:
                raise ValueError(f"Duplicate document ID {key!r} in {path}")
            records[key] = row
    return records


def quality_for(synthetic, audit):
    return (
        synthetic.get("reconstruction_quality")
        or audit.get("reconstruction_quality")
        or {}
    )


def selected_by_default(quality):
    if "selected" in quality:
        return bool(quality["selected"])
    return bool(
        quality.get("eligible")
        and quality.get("quality_score", 0) >= 85
        and not quality.get("reasons")
    )


def page_name(doc_id):
    return f"document_{quote(str(doc_id), safe='-_')}.html"


def summary_rows(source_rows, audits, synthetic_rows):
    rows = []
    for doc_id, source in source_rows.items():
        audit = audits.get(doc_id)
        if audit is None:
            raise ValueError(f"Document {doc_id!r} is missing from the links audit")
        synthetic = synthetic_rows.get(doc_id, {})
        quality = quality_for(synthetic, audit)
        reasons = quality.get("reasons") or audit.get("review_reasons") or []
        rows.append({
            "doc_id": doc_id,
            "case_type": str(source.get("case_type") or "unknown"),
            "doc_type": str(source.get("doc_type") or "unknown"),
            "score": quality.get("quality_score", "n/a"),
            "tier": quality.get("quality_tier", "unassessed"),
            "selected": selected_by_default(quality),
            "replacements": quality.get(
                "replacements",
                (synthetic.get("reconstruction_stats") or {}).get("replacements", 0),
            ),
            "reasons": list(reasons),
        })
    return rows


def render_index(rows, output_name):
    counts = Counter(row["tier"] for row in rows)
    selected = sum(row["selected"] for row in rows)
    table_rows = []
    for row in sorted(rows, key=lambda item: (item["tier"], str(item["score"]), item["doc_id"])):
        reasons = ", ".join(row["reasons"])
        score = html.escape(str(row["score"]))
        tier = html.escape(str(row["tier"]))
        tier_class = html.escape(str(row["tier"]).lower().replace(" ", "-"), quote=True)
        table_rows.append(
            f"<tr data-search='{html.escape(' '.join([row['doc_id'], row['case_type'], row['doc_type'], row['tier'], reasons]), quote=True)}'>"
            f"<td><a href='{html.escape(page_name(row['doc_id']), quote=True)}'>{html.escape(row['doc_id'])}</a></td>"
            f"<td>{html.escape(row['case_type'])}</td>"
            f"<td>{html.escape(row['doc_type'])}</td>"
            f"<td class='score {tier_class}'>{score}</td>"
            f"<td class='tier {tier_class}'>{tier}</td>"
            f"<td>{'yes' if row['selected'] else 'no'}</td>"
            f"<td>{html.escape(str(row['replacements']))}</td>"
            f"<td title='{html.escape(reasons, quote=True)}'>{html.escape(reasons or 'none')}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>VDT demo review</title>
<style>
body {{ margin: 24px; font-family: Arial, sans-serif; color: #24292f; }}
h1 {{ margin-bottom: 8px; }}
.note {{ color: #57606a; }}
.stats {{ background: #fff8c5; border: 1px solid #d4a72c; border-radius: 6px; padding: 12px; margin: 16px 0; }}
input {{ width: min(720px, 100%); padding: 10px; border: 1px solid #8c959f; border-radius: 6px; font-size: 15px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13px; }}
th, td {{ border: 1px solid #d0d7de; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; position: sticky; top: 0; }}
.score.clean, .tier.clean {{ color: #116329; font-weight: 700; }}
.score.review, .tier.review {{ color: #9a6700; font-weight: 700; }}
.score.reject, .tier.reject {{ color: #cf222e; font-weight: 700; }}
.table-wrap {{ overflow: auto; max-height: 78vh; }}
</style>
</head>
<body>
<h1>VDT-Anonymization demo review</h1>
<p class='note'>Open a document ID to inspect the original and synthetic text side by side. Search accepts an ID, case type, document type, tier, or review reason.</p>
<div class='stats'>Documents: {len(rows)} | Selected: {selected} | Tiers: {html.escape(str(dict(counts)))}<br>Generated pages: {html.escape(output_name)}</div>
<label for='search'>Filter documents</label><br>
<input id='search' type='search' placeholder='e.g. ambiguous_identity, reject, hon_nhan_gia_dinh, 1000213'>
<div class='table-wrap'>
<table id='documents'>
<thead><tr><th>Document</th><th>Case type</th><th>Document type</th><th>Score</th><th>Tier</th><th>Selected</th><th>Replacements</th><th>Reasons</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</div>
<script>
const search = document.getElementById('search');
search.addEventListener('input', () => {{
  const query = search.value.trim().toLowerCase();
  document.querySelectorAll('#documents tbody tr').forEach(row => {{
    row.hidden = query && !row.dataset.search.toLowerCase().includes(query);
  }});
}});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", default=DEFAULT_SOURCE)
    parser.add_argument("--links-file", default=DEFAULT_LINKS)
    parser.add_argument("--synthetic-file", default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N documents; 0 means all")
    args = parser.parse_args()

    for path in (args.source_file, args.links_file, args.synthetic_file):
        if not os.path.exists(path):
            parser.error(f"Input file does not exist: {path}")
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    source_rows = load_jsonl(args.source_file)
    audits = load_jsonl(args.links_file, id_field="doc_id")
    synthetic_rows = load_jsonl(args.synthetic_file)
    rows = summary_rows(source_rows, audits, synthetic_rows)
    if args.limit:
        rows = rows[:args.limit]
        source_rows = {row["doc_id"]: source_rows[row["doc_id"]] for row in rows}

    os.makedirs(args.output_dir, exist_ok=True)
    for row in rows:
        doc_id = row["doc_id"]
        with open(os.path.join(args.output_dir, page_name(doc_id)), "w", encoding="utf-8") as handle:
            handle.write(build_html(doc_id, source_rows[doc_id], audits[doc_id], synthetic_rows.get(doc_id)))
    index_path = os.path.join(args.output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(render_index(rows, args.output_dir))
    print(f"[INSPECT-DEMO] Documents: {len(rows)}")
    print(f"[INSPECT-DEMO] Review folder: {os.path.abspath(args.output_dir)}")
    print(f"[INSPECT-DEMO] Open: {os.path.abspath(index_path)}")


if __name__ == "__main__":
    main()
