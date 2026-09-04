"""Render one complete legal document with NER/linking annotations.

The output is a standalone HTML file and requires no third-party package or
network access. It highlights the original document, while the entity table
shows the linked entity ID, marker, synthetic replacement, and evidence.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter


LABEL_COLORS = {
    "PER": ("#ffe08a", "#5c4300"),
    "LOC": ("#a8d8ff", "#063b5c"),
}


def document_id(row, row_number):
    return str(row.get("case_id", row.get("doc_name", row.get("id", row_number))))


def find_jsonl_row(path, target_id):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if document_id(row, row_number) == str(target_id):
                return row
    return None


def find_audit_row(path, target_id):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("doc_id")) == str(target_id):
                return row
    return None


def make_spans(audit_row, text_length):
    spans = []
    for entity in audit_row.get("entities", []):
        for mention in entity.get("mentions", []):
            start = max(0, min(text_length, int(mention.get("start", 0))))
            end = max(start, min(text_length, int(mention.get("end", start))))
            if end <= start:
                continue
            spans.append({
                "start": start,
                "end": end,
                "label": entity.get("label", ""),
                "entity_id": entity.get("entity_id", ""),
                "marker": entity.get("marker"),
                "synthetic_value": entity.get("synthetic_value"),
                "score": mention.get("score"),
                "detectors": mention.get("detectors", []),
                "link_evidence": mention.get("link_evidence"),
                "mention_text": mention.get("text", ""),
            })
    return spans


def span_priority(span):
    score = span.get("score")
    return (
        int(bool(span.get("marker"))),
        span["end"] - span["start"],
        float(score) if isinstance(score, (int, float)) else -1.0,
    )


def render_document(text, spans):
    boundaries = {0, len(text)}
    for span in spans:
        boundaries.add(span["start"])
        boundaries.add(span["end"])
    boundaries = sorted(boundaries)
    rendered = []
    conflict_segments = 0

    for left, right in zip(boundaries, boundaries[1:]):
        if left == right:
            continue
        active = [
            span for span in spans
            if span["start"] <= left and span["end"] >= right
        ]
        if not active:
            rendered.append(html.escape(text[left:right]))
            continue

        active.sort(key=span_priority, reverse=True)
        selected = active[0]
        if len(active) > 1:
            conflict_segments += 1
        label = selected["label"]
        background, foreground = LABEL_COLORS.get(label, ("#e5e7eb", "#111827"))
        score = selected.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        title = (
            f"{label} | entity={selected['entity_id']} | "
            f"marker={selected.get('marker') or 'unmarked'} | "
            f"synthetic={selected.get('synthetic_value') or 'unchanged'} | "
            f"confidence={score_text} | "
            f"evidence={selected.get('link_evidence') or 'n/a'}"
        )
        rendered.append(
            f"<mark class='entity entity-{html.escape(label.lower())}' "
            f"style='background:{background};color:{foreground}' "
            f"title='{html.escape(title, quote=True)}'>"
            f"{html.escape(text[left:right])}</mark>"
        )

    return "".join(rendered), conflict_segments


def entity_table(audit_row):
    rows = []
    for entity in audit_row.get("entities", []):
        mentions = entity.get("mentions", [])
        first_text = mentions[0].get("text", "") if mentions else ""
        scores = [m.get("score") for m in mentions if isinstance(m.get("score"), (int, float))]
        score = f"{sum(scores) / len(scores):.3f}" if scores else "n/a"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entity.get('entity_id', '')))}</td>"
            f"<td>{html.escape(str(entity.get('label', '')))}</td>"
            f"<td>{html.escape(str(entity.get('marker') or 'unmarked'))}</td>"
            f"<td>{html.escape(str(first_text))}</td>"
            f"<td>{html.escape(str(entity.get('synthetic_value') or 'unchanged'))}</td>"
            f"<td>{len(mentions)}</td>"
            f"<td>{score}</td>"
            f"<td>{html.escape(str(entity.get('name_rule') or ''))}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_html(doc_id, source_row, audit_row, synthetic_row=None):
    source_text = str(source_row.get("markdown", source_row.get("text", "")))
    spans = make_spans(audit_row, len(source_text))
    highlighted, conflict_segments = render_document(source_text, spans)
    label_counts = Counter(entity.get("label") for entity in audit_row.get("entities", []))
    marker_entities = sum(bool(entity.get("marker")) for entity in audit_row.get("entities", []))
    synthetic_text = ""
    if synthetic_row:
        synthetic_text = str(synthetic_row.get("synthetic_markdown", ""))

    synthetic_section = ""
    if synthetic_text:
        synthetic_section = (
            "<h2>Synthetic reconstruction</h2>"
            "<p class='note'>This text is shown without highlights because its character offsets change after replacement.</p>"
            f"<pre class='document'>{html.escape(synthetic_text)}</pre>"
        )

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Entity inspection: {html.escape(str(doc_id))}</title>
<style>
body {{ margin: 24px; font-family: Arial, sans-serif; color: #24292f; }}
h1, h2 {{ margin-bottom: 8px; }}
.note {{ color: #57606a; }}
.legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
.badge {{ border-radius: 5px; padding: 4px 8px; font-weight: 700; }}
.document {{ white-space: pre-wrap; overflow-wrap: anywhere; font: 14px/1.7 Consolas, monospace; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; max-height: 75vh; overflow: auto; }}
.entity {{ border-radius: 3px; padding: 1px 2px; cursor: help; }}
.stats {{ background: #fff8c5; border: 1px solid #d4a72c; border-radius: 6px; padding: 10px; }}
details {{ margin-top: 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; position: sticky; top: 0; }}
.table-wrap {{ max-height: 60vh; overflow: auto; }}
</style>
</head>
<body>
<h1>Entity inspection: document {html.escape(str(doc_id))}</h1>
<div class='stats'>
  Entities: {len(audit_row.get('entities', []))} |
  Mentions: {len(spans)} |
  Marker-based entities: {marker_entities} |
  Labels: {html.escape(str(dict(label_counts)))} |
  Overlap segments resolved for display: {conflict_segments}
</div>
<div class='legend'>
  <span class='badge' style='background:#ffe08a;color:#5c4300'>PER</span>
  <span class='badge' style='background:#a8d8ff;color:#063b5c'>LOC</span>
  <span>Hover a highlight to see entity ID, marker, confidence, and replacement.</span>
</div>
<h2>Complete original document</h2>
<pre class='document'>{highlighted}</pre>
{synthetic_section}
<details>
<summary><b>Linked entity table</b></summary>
<div class='table-wrap'>
<table>
<thead><tr><th>Entity ID</th><th>Label</th><th>Marker</th><th>First mention</th><th>Synthetic value</th><th>Mentions</th><th>Avg. confidence</th><th>Name rule</th></tr></thead>
<tbody>{entity_table(audit_row)}</tbody>
</table>
</div>
</details>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render one full document with NER/entity-link annotations")
    parser.add_argument("--source-file", required=True, help="Original legal JSONL")
    parser.add_argument("--links-file", required=True, help="entity_links.jsonl")
    parser.add_argument("--synthetic-file", default=None, help="Optional synthetic_unanonymized.jsonl")
    parser.add_argument("--doc-id", default=None, help="Document ID; defaults to the first audit record")
    parser.add_argument("--output-file", required=True, help="HTML output path")
    args = parser.parse_args()

    if not os.path.exists(args.source_file):
        parser.error(f"Source file does not exist: {args.source_file}")
    if not os.path.exists(args.links_file):
        parser.error(f"Links file does not exist: {args.links_file}")

    target_id = args.doc_id
    if target_id is None:
        with open(args.links_file, "r", encoding="utf-8-sig") as handle:
            first = next((json.loads(line) for line in handle if line.strip()), None)
        if not first:
            parser.error("Links file is empty")
        target_id = first.get("doc_id")

    source_row = find_jsonl_row(args.source_file, target_id)
    audit_row = find_audit_row(args.links_file, target_id)
    if source_row is None:
        parser.error(f"Document {target_id} was not found in {args.source_file}")
    if audit_row is None:
        parser.error(f"Document {target_id} was not found in {args.links_file}")

    synthetic_row = None
    if args.synthetic_file:
        if not os.path.exists(args.synthetic_file):
            parser.error(f"Synthetic file does not exist: {args.synthetic_file}")
        synthetic_row = find_jsonl_row(args.synthetic_file, target_id)

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)
    source_text = str(source_row.get("markdown", source_row.get("text", "")))
    with open(args.output_file, "w", encoding="utf-8") as handle:
        handle.write(build_html(target_id, source_row, audit_row, synthetic_row))

    print(f"[INSPECT] Document: {target_id}")
    print(f"[INSPECT] HTML saved to: {args.output_file}")
    print(f"[INSPECT] Source characters: {len(source_text)}")


if __name__ == "__main__":
    main()
