"""Create a compact HTML inspection report for a raw-name baseline."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from .dataset import normalize_text


def prediction(pair: dict) -> int:
    a, b = pair["mention_a"], pair["mention_b"]
    return int(
        a.get("label") == b.get("label")
        and normalize_text(a.get("surface")) == normalize_text(b.get("surface"))
    )


def status(pair: dict) -> str:
    target, predicted = int(pair["target_linked"]), prediction(pair)
    if target and predicted:
        return "true-positive"
    if target and not predicted:
        return "missed-link"
    if not target and predicted:
        return "false-merge"
    return "true-negative"


def compact_context(value: object, limit: int = 520) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def load_examples(path: Path, per_status: int) -> tuple[Counter, list[dict]]:
    counts = Counter()
    examples = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            label = status(pair)
            counts[label] += 1
            if label != "true-negative" and sum(item["status"] == label for item in examples) < per_status:
                examples.append({
                    "status": label,
                    "doc_id": pair.get("doc_id", ""),
                    "difficulty": pair.get("difficulty", ""),
                    "target": int(pair["target_linked"]),
                    "predicted": prediction(pair),
                    "surface_a": pair["mention_a"].get("surface", ""),
                    "surface_b": pair["mention_b"].get("surface", ""),
                    "context_a": compact_context(pair["mention_a"].get("context")),
                    "context_b": compact_context(pair["mention_b"].get("context")),
                })
    return counts, examples


def render(path: Path, counts: Counter, examples: list[dict]) -> str:
    total = sum(counts.values())
    rows = []
    for item in examples:
        cls = html.escape(item["status"])
        rows.append(
            f"<article class='example {cls}' data-status='{cls}'>"
            f"<header><strong>{html.escape(item['status'].replace('-', ' '))}</strong>"
            f"<span>doc {html.escape(str(item['doc_id']))} · {html.escape(str(item['difficulty']))}</span></header>"
            f"<div class='mentions'><div><b>A: {html.escape(str(item['surface_a']))}</b>"
            f"<p>{html.escape(item['context_a'])}</p></div>"
            f"<div><b>B: {html.escape(str(item['surface_b']))}</b>"
            f"<p>{html.escape(item['context_b'])}</p></div></div></article>"
        )
    cards = []
    for key, title in (("true-positive", "Correct links"), ("missed-link", "Missed links"), ("false-merge", "False merges")):
        value = counts[key]
        width = (value / total * 100) if total else 0
        cards.append(
            f"<div class='card'><span>{title}</span><strong>{value:,}</strong>"
            f"<div class='bar'><i style='width:{width:.2f}%'></i></div></div>"
        )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Raw linker baseline inspection</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#172033;background:#f6f8fb}}
h1{{margin-bottom:4px}} .note{{color:#5c667a}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}
.card,.example{{background:white;border:1px solid #dce2eb;border-radius:8px;padding:14px;box-shadow:0 1px 2px #0000000b}}
.card span{{display:block;color:#5c667a;font-size:13px}} .card strong{{font-size:26px;display:block;margin:5px 0}}
.bar{{height:7px;background:#e9edf3;border-radius:4px;overflow:hidden}} .bar i{{display:block;height:100%;background:#4978d1}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}} button{{border:1px solid #bbc6d6;background:white;border-radius:5px;padding:7px 12px;cursor:pointer}}
.example{{margin:12px 0}} .example[hidden]{{display:none}} header{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:10px}}
header strong{{text-transform:capitalize}} header span{{color:#667085;font-size:13px}} .mentions{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.mentions>div{{background:#f7f9fc;border:1px solid #e6eaf0;padding:10px;border-radius:5px}} p{{white-space:pre-wrap;font:13px/1.5 Consolas,monospace;margin:8px 0 0;overflow-wrap:anywhere}}
.false-merge header strong{{color:#b42318}} .missed-link header strong{{color:#b54708}} .true-positive header strong{{color:#087443}}
@media(max-width:700px){{.cards,.mentions{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Raw-name baseline inspection</h1>
<p class='note'>Exact normalized surface matching on the faithful test split. A false merge predicts two different entities are the same; a missed link fails to join two mentions of one entity.</p>
<section class='cards'>{''.join(cards)}</section>
<div class='controls'><button data-filter='all'>All examples</button><button data-filter='true-positive'>Correct links</button><button data-filter='missed-link'>Missed links</button><button data-filter='false-merge'>False merges</button></div>
<main id='examples'>{''.join(rows)}</main>
<script>document.querySelectorAll('button[data-filter]').forEach(button=>button.addEventListener('click',()=>{{const f=button.dataset.filter;document.querySelectorAll('.example').forEach(x=>x.hidden=f!='all'&&x.dataset.status!=f)}}));</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-status", type=int, default=12)
    args = parser.parse_args()
    counts, examples = load_examples(args.pairs.resolve(), args.per_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.pairs, counts, examples), encoding="utf-8")
    print(json.dumps({"pairs": sum(counts.values()), "counts": counts, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
