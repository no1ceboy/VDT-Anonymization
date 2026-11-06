"""Build a standalone V4 report for the exact-text and location-rule baseline."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..entity_linking.baseline import metrics, predict


DEFAULT_TEST_PAIRS = Path("outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test.jsonl")
DEFAULT_CHALLENGE_PAIRS = Path(
    "outputs/entity_linking_v4_location_hierarchy_20261009/pairs/test_challenge.jsonl"
)
DEFAULT_OUTPUT = Path("outputs/entity_linker_v4_ablation_20261009/baseline_cases.html")

CHALLENGE_GROUPS = (
    "collision_donor_positive",
    "collision_recipient_positive",
    "collision_same_surface_negative",
)
NATURAL_GROUPS = (
    "surface_true_positive",
    "surface_true_negative",
    "surface_missed_link",
    "surface_false_link",
)
GROUP_TITLES = {
    "collision_donor_positive": "Collision: donor mention should link",
    "collision_recipient_positive": "Collision: recipient mention should link",
    "collision_same_surface_negative": "Collision: same text, different entities",
    "surface_true_positive": "Natural test: correct link",
    "surface_true_negative": "Natural test: correct separation",
    "surface_missed_link": "Natural test: missed link",
    "surface_false_link": "Natural test: false link",
}
GROUP_EXPLANATIONS = {
    "collision_donor_positive": (
        "The mentions belong to the donor person. The baseline misses this because the two "
        "mention strings differ."
    ),
    "collision_recipient_positive": (
        "The overwritten alias and the recipient's other mention still refer to one person, "
        "but their strings differ."
    ),
    "collision_same_surface_negative": (
        "The mentions have the same string after the controlled edit but belong to different "
        "weak-map identities."
    ),
    "surface_true_positive": "The normalized strings match and the weak map assigns one identity.",
    "surface_true_negative": "The strings differ and the weak map assigns separate identities.",
    "surface_missed_link": "The weak map assigns one identity, but the mention strings differ.",
    "surface_false_link": "The strings match, but the weak map assigns different identities.",
}
DIFFICULTY_LABELS = {
    "positive": "Positive pair",
    "same_label": "Same NER type; different text",
    "same_label_location_conflict": "Same NER type; conflicting location path",
    "same_label_name_overlap": "Same NER type; overlapping name tokens",
    "same_label_surface": "Same text; different weak identities",
    "collision_donor_positive": "Collision: donor should link",
    "collision_recipient_positive": "Collision: recipient should link",
    "collision_same_surface_negative": "Collision: same text, keep separate",
}
CHALLENGE_LABELS = {
    "address_marker": "Location abbreviations",
    "alias_language": "Name and alias variation",
    "dotted_initial": "Dotted initials",
    "full_name_marker": "Full names and partial initials",
    "numbered_marker": "Numbered initials",
    "numbered_multi": "Multi-character numbered initials",
    "organization_marker": "Organization abbreviations",
    "person_context": "Person-context cases",
    "Unknown": "Unspecified challenge",
}


def _outcome(target: int, prediction: int) -> str:
    if target == prediction:
        return "correct"
    return "false_link" if prediction else "missed_link"


def _natural_group(target: int, prediction: int) -> str:
    if target and prediction:
        return "surface_true_positive"
    if target and not prediction:
        return "surface_missed_link"
    if not target and prediction:
        return "surface_false_link"
    return "surface_true_negative"


def _compact_pair(pair: dict, source: str, group: str, prediction: int,
                  location_decision: str | None) -> dict:
    mentions = {}
    for name in ("mention_a", "mention_b"):
        mention = pair[name]
        mentions[name] = {
            "label": str(mention.get("label") or "Unknown"),
            "surface": str(mention.get("surface") or ""),
            "context": str(mention.get("context") or ""),
        }
    target = int(pair["target_linked"])
    return {
        "source": source,
        "group": group,
        "pair_id": str(pair.get("pair_id") or ""),
        "doc_id": str(pair.get("doc_id") or ""),
        "difficulty": str(pair.get("difficulty") or "Unknown"),
        "input_variant": str(pair.get("input_variant") or ""),
        "target": target,
        "prediction": prediction,
        "outcome": _outcome(target, prediction),
        "location_decision": location_decision or "deferred",
        "mentions": mentions,
    }


def analyze_pairs(path: Path, source: str, per_group: int) -> dict:
    total = Counter()
    by_difficulty: dict[str, Counter] = defaultdict(Counter)
    by_primary_challenge: dict[str, Counter] = defaultdict(Counter)
    location_counts = Counter()
    pair_count = 0
    selected: dict[str, list[dict]] = defaultdict(list)
    selected_docs: dict[str, set[str]] = defaultdict(set)

    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            pair_count += 1
            target = int(pair["target_linked"])
            prediction, location_decision = predict(pair)
            key = "tp" if target and prediction else "fn" if target else "fp" if prediction else "tn"
            total[key] += 1
            by_difficulty[str(pair.get("difficulty") or "Unknown")][key] += 1

            metadata = pair.get("document_metadata") or {}
            by_primary_challenge[str(metadata.get("primary_challenge") or "Unknown")][key] += 1
            if location_decision:
                location_counts[location_decision] += 1

            if source == "challenge":
                group = str(pair.get("difficulty") or "")
                if group not in CHALLENGE_GROUPS:
                    continue
            else:
                group = _natural_group(target, prediction)

            if len(selected[group]) >= per_group:
                continue
            doc_id = str(pair.get("doc_id") or pair.get("pair_id") or "")
            if doc_id and doc_id in selected_docs[group]:
                continue
            selected[group].append(_compact_pair(pair, source, group, prediction, location_decision))
            if doc_id:
                selected_docs[group].add(doc_id)

    if not pair_count:
        raise ValueError(f"No pairs found in {path}")

    ordered_groups = CHALLENGE_GROUPS if source == "challenge" else NATURAL_GROUPS
    examples = [case for group in ordered_groups for case in selected.get(group, [])]
    return {
        "source": source,
        "path": str(path),
        "pairs": pair_count,
        "overall": metrics(total),
        "by_difficulty": {name: metrics(values) for name, values in sorted(by_difficulty.items())},
        "by_primary_challenge": {
            name: metrics(values) for name, values in sorted(by_primary_challenge.items())
        },
        "location_counts": {
            "link": location_counts["link"],
            "block": location_counts["block"],
            "deferred": pair_count - location_counts["link"] - location_counts["block"],
        },
        "examples": examples,
    }


def _context_html(value: str) -> str:
    open_tag, close_tag = "[MENTION]", "[/MENTION]"
    start = value.find(open_tag)
    end = value.find(close_tag, start + len(open_tag)) if start >= 0 else -1
    if start >= 0 and end >= 0:
        left = max(0, start - 125)
        right = min(len(value), end + len(close_tag) + 125)
        value = ("..." if left else "") + value[left:right] + ("..." if right < len(value) else "")
    escaped = html.escape(value)
    return escaped.replace(open_tag, "<mark>").replace(close_tag, "</mark>")


def _metric_cells(result: dict) -> str:
    confusion = result["confusion"]
    return (
        f"<td>{result['examples']:,}</td><td>{result['accuracy']:.3f}</td>"
        f"<td>{result['precision']:.3f}</td><td>{result['recall']:.3f}</td>"
        f"<td>{result['f1']:.3f}</td>"
        f"<td>{confusion['tp']:,} / {confusion['fp']:,} / "
        f"{confusion['tn']:,} / {confusion['fn']:,}</td>"
    )


def _render_breakdown(title: str, values: dict) -> str:
    rows = []
    for name, result in values.items():
        label = DIFFICULTY_LABELS.get(name, CHALLENGE_LABELS.get(name, name))
        rows.append(
            f"<tr><th scope='row'>{html.escape(label)}</th>{_metric_cells(result)}</tr>"
        )
    return (
        f"<h3>{html.escape(title)}</h3><div class='table-scroll'><table>"
        "<thead><tr><th>Case type</th><th>Pairs</th><th>Accuracy</th><th>Precision</th>"
        "<th>Recall</th><th>F1</th><th>TP / FP / TN / FN</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_case(case: dict) -> str:
    target = case["target"]
    prediction = case["prediction"]
    outcome = case["outcome"]
    target_text = "LINK" if target else "KEEP SEPARATE"
    prediction_text = "LINK" if prediction else "KEEP SEPARATE"
    target_class = "link" if target else "separate"
    mentions = []
    for name, label in (("mention_a", "Mention A"), ("mention_b", "Mention B")):
        mention = case["mentions"][name]
        mentions.append(
            "<section class='mention'>"
            f"<div class='mention-heading'><strong>{label}</strong>"
            f"<span>{html.escape(mention['label'])}</span></div>"
            f"<div class='surface'>{html.escape(mention['surface'])}</div>"
            f"<pre>{_context_html(mention['context'])}</pre></section>"
        )

    explanation = GROUP_EXPLANATIONS[case["group"]]
    searchable = " ".join((
        case["doc_id"], case["pair_id"], case["difficulty"],
        case["mentions"]["mention_a"]["surface"], case["mentions"]["mention_b"]["surface"],
    ))
    return (
        f"<article class='case' data-source='{case['source']}' "
        f"data-outcome='{outcome}' data-search='{html.escape(searchable, quote=True).casefold()}'>"
        "<header class='case-header'><div>"
        f"<span class='eyebrow'>{'Collision challenge' if case['source'] == 'challenge' else 'Natural test'}</span>"
        f"<h3>{html.escape(GROUP_TITLES[case['group']])}</h3>"
        f"<p>{html.escape(explanation)}</p></div>"
        f"<span class='target {target_class}'>Weak-map target: {target_text}</span></header>"
        "<div class='case-meta'>"
        f"<span>Document {html.escape(case['doc_id'])}</span>"
        f"<span>Pair {html.escape(case['pair_id'])}</span>"
        f"<span>Location rule: {html.escape(case['location_decision'])}</span>"
        + (f"<span>Variant {html.escape(case['input_variant'])}</span>" if case["input_variant"] else "")
        + "</div>"
        "<div class='prediction-row'><strong>Baseline:</strong> "
        f"<span class='{outcome}'>{prediction_text} - {outcome.replace('_', ' ')}</span></div>"
        f"<div class='mentions'>{''.join(mentions)}</div></article>"
    )


def render_html(test: dict, challenge: dict) -> str:
    summary_rows = []
    for title, result in (("Natural test", test["overall"]),
                          ("Collision challenge", challenge["overall"])):
        summary_rows.append(
            f"<tr><th scope='row'>{title}</th>{_metric_cells(result)}</tr>"
        )
    cases = "".join(_render_case(case) for case in test["examples"] + challenge["examples"])
    example_count = len(test["examples"]) + len(challenge["examples"])
    test_location = test["location_counts"]
    challenge_location = challenge["location_counts"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V4 exact-text and location baseline</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#596673; --line:#d8e0e7; --paper:#fff; --page:#f3f6f8; --blue:#145b8f; --green:#176b48; --red:#a73737; --amber:#825609; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:24px; background:var(--page); color:var(--ink); font:15px/1.5 "Segoe UI",Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ margin:0 0 4px; font-size:1.7rem; }}
h2 {{ font-size:1.2rem; margin:22px 0 8px; }}
h3 {{ font-size:1.05rem; margin:20px 0 8px; }}
.intro,.note {{ color:var(--muted); margin:4px 0 16px; }}
.warning {{ border-left:4px solid var(--amber); background:#fff7df; padding:12px 14px; margin:16px 0; }}
.table-scroll {{ overflow-x:auto; background:var(--paper); border:1px solid var(--line); border-radius:8px; margin:8px 0 18px; }}
table {{ border-collapse:collapse; min-width:760px; width:100%; }}
th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }}
thead th {{ background:#eaf0f4; }}
tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
.meta-line {{ color:var(--muted); margin:8px 0 18px; }}
.controls {{ display:flex; align-items:end; flex-wrap:wrap; gap:12px; margin:20px 0 10px; }}
label {{ display:grid; gap:4px; color:var(--muted); font-size:.9rem; }}
select,input {{ min-height:40px; padding:7px 10px; color:var(--ink); background:var(--paper); border:1px solid #aebbc6; border-radius:5px; font:inherit; }}
input[type=search] {{ width:min(360px,90vw); }}
#visibleCount {{ color:var(--muted); margin:4px 0 12px; }}
.case {{ background:var(--paper); border:1px solid var(--line); border-radius:9px; padding:16px; margin:14px 0; box-shadow:0 1px 2px #17212b0c; }}
.case-header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
.eyebrow {{ color:var(--blue); font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }}
.case-header h3 {{ margin:2px 0 3px; }}
.case-header p {{ color:var(--muted); margin:4px 0 8px; }}
.target {{ flex:0 0 auto; padding:5px 9px; border-radius:5px; font-weight:700; }}
.target.link {{ background:#e5f5eb; color:var(--green); }} .target.separate {{ background:#edf1f4; color:#334453; }}
.case-meta {{ display:flex; flex-wrap:wrap; gap:7px 16px; margin:6px 0 12px; color:var(--muted); font-size:.85rem; }}
.prediction-row {{ border-top:2px solid var(--line); padding:8px; margin:8px 0; }}
.correct {{ color:var(--green); }} .false_link,.missed_link {{ color:var(--red); }}
.mentions {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
.mention {{ min-width:0; border-top:1px solid var(--line); padding-top:8px; }}
.mention-heading {{ display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:.85rem; }}
.surface {{ font-weight:700; margin:4px 0; overflow-wrap:anywhere; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.55 Consolas,"Cascadia Mono",monospace; background:#f5f7f9; border-radius:5px; margin:0; padding:10px; }}
mark {{ background:#ffdf81; color:#302100; border-radius:2px; padding:0 2px; }}
@media(max-width:760px) {{ body {{ padding:14px; }} .case-header {{ display:block; }} .target {{ display:inline-block; margin-top:5px; }} .mentions {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main>
  <h1>V4 rule baseline</h1>
  <p class="intro">One practical heuristic: apply high-confidence location rules; for other mentions, link only when NER labels and normalized text match exactly.</p>
  <div class="warning"><strong>Scope:</strong> V4 uses source documents labeled <code>original_anonymized_markdown</code>, with weak reconstruction-map identity labels and no human annotation. Its collision challenge is deliberately constructed. These numbers are not accuracy estimates for deployment on independently labeled, un-anonymized documents.</div>
  <h2>Overall results</h2>
  <div class="table-scroll"><table>
    <thead><tr><th>Evaluation split</th><th>Pairs</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / TN / FN</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table></div>
  <p class="meta-line">Location rule decisions — natural test: {test_location['link']:,} links, {test_location['block']:,} separations, {test_location['deferred']:,} deferred; collision challenge: {challenge_location['link']:,} links, {challenge_location['block']:,} separations, {challenge_location['deferred']:,} deferred. Showing {example_count} readable examples.</p>
  {_render_breakdown('Natural test by pair difficulty', test['by_difficulty'])}
  {_render_breakdown('Natural test by challenge family', test['by_primary_challenge'])}
  {_render_breakdown('Collision challenge by scenario', challenge['by_difficulty'])}
  <h2>Inspect representative pairs</h2>
  <div class="controls">
    <label>Split<select id="splitFilter"><option value="all">Both splits</option><option value="test">Natural test</option><option value="challenge">Collision challenge</option></select></label>
    <label>Outcome<select id="outcomeFilter"><option value="all">All outcomes</option><option value="false_link">False link</option><option value="missed_link">Missed link</option><option value="correct">Correct</option></select></label>
    <label>Find case<input id="caseSearch" type="search" placeholder="Document, mention, or case type"></label>
  </div>
  <p id="visibleCount" aria-live="polite"></p>
  <section id="cases" aria-label="Representative baseline examples">{cases}</section>
</main>
<script>
const cards = [...document.querySelectorAll('.case')];
const splitFilter = document.getElementById('splitFilter');
const outcomeFilter = document.getElementById('outcomeFilter');
const caseSearch = document.getElementById('caseSearch');
const visibleCount = document.getElementById('visibleCount');
function applyFilters() {{
  const split = splitFilter.value;
  const outcome = outcomeFilter.value;
  const query = caseSearch.value.trim().toLocaleLowerCase();
  let visible = 0;
  for (const card of cards) {{
    const keep = (split === 'all' || card.dataset.source === split)
      && (outcome === 'all' || card.dataset.outcome === outcome)
      && (!query || card.dataset.search.includes(query));
    card.hidden = !keep;
    if (keep) visible += 1;
  }}
  visibleCount.textContent = `Showing ${{visible}} of ${{cards.length}} sampled cases`;
}}
splitFilter.addEventListener('change', applyFilters);
outcomeFilter.addEventListener('change', applyFilters);
caseSearch.addEventListener('input', applyFilters);
applyFilters();
</script>
</body>
</html>
"""


def build_review(test_path: Path, challenge_path: Path, output_path: Path,
                 per_group: int = 2) -> dict:
    if per_group < 1:
        raise ValueError("per_group must be at least 1")
    test = analyze_pairs(test_path, "test", per_group)
    challenge = analyze_pairs(challenge_path, "challenge", per_group)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(test, challenge), encoding="utf-8")
    return {"test": test, "challenge": challenge, "output": output_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-pairs", type=Path, default=DEFAULT_TEST_PAIRS)
    parser.add_argument("--challenge-pairs", type=Path, default=DEFAULT_CHALLENGE_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-group", type=int, default=2,
                        help="representative pairs per baseline outcome/scenario")
    args = parser.parse_args()
    for path in (args.test_pairs, args.challenge_pairs):
        if not path.is_file():
            parser.error(f"pair file does not exist: {path}")
    result = build_review(args.test_pairs, args.challenge_pairs, args.output, args.per_group)
    print(f"Natural test pairs: {result['test']['pairs']:,}")
    print(f"Collision challenge pairs: {result['challenge']['pairs']:,}")
    print(f"Representative cases: {len(result['test']['examples']) + len(result['challenge']['examples'])}")
    print(f"HTML: {result['output'].resolve()}")


if __name__ == "__main__":
    main()
