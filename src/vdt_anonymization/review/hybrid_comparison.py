"""Compare the hybrid scorer with saved baselines and render real held-out cases."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from ..entity_linking.clustering import complete_link_clusters
from ..entity_linking.dataset import normalize_text
from ..entity_linking.hybrid import HybridPairScorer, _document_rows, _read_jsonl
from ..entity_linking.location_constraints import location_rule_decision
from ..entity_linking.raw_dataset import raw_pair_features
from ..entity_linking.report_baseline import evaluate as evaluate_report_baseline


COLORS = ["#d6ebff", "#ffe1bc", "#d5f2df", "#e9dcff", "#ffd7e2", "#d8f1ef",
          "#f3e9ba", "#d7defa", "#e0edc4", "#efdbce", "#cde6fb", "#f4d1c5"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_pair(metric: dict) -> dict:
    return metric.get("overall") or metric["metrics"]["overall"]


def _report_pair_prediction(owners: dict, row: dict) -> int:
    a, b = row["mention_a"], row["mention_b"]
    left = (int(a["start"]), int(a["end"]))
    right = (int(b["start"]), int(b["end"]))
    if left not in owners or right not in owners:
        raise ValueError(f"Pair {row.get('pair_id')} is missing from the report-baseline mention map")
    return int(owners[left] == owners[right])


def _exact_owner(document: dict) -> dict[tuple[int, int], int]:
    mentions = document["mentions"]
    scores, must_link, cannot_link = {}, set(), set()
    for left, right in combinations(range(len(mentions)), 2):
        a, b = mentions[left], mentions[right]
        if str(a.get("label") or "").casefold() != str(b.get("label") or "").casefold():
            continue
        pair = {"mention_a": a, "mention_b": b}
        location = location_rule_decision(pair)
        if location == "link":
            must_link.add((left, right))
        elif location == "block":
            cannot_link.add((left, right))
        elif normalize_text(a.get("surface")) == normalize_text(b.get("surface")):
            scores[(left, right)] = 1.0
    clusters = complete_link_clusters(mentions, scores, 0.5, cannot_link, must_link)
    return {
        (int(mentions[index]["start"]), int(mentions[index]["end"])): cluster_id
        for cluster_id, cluster in enumerate(clusters)
        for index in cluster
    }


def _render_document(document: dict, owners: dict[tuple[int, int], int]) -> str:
    text = document["text"]
    mentions = document["mentions"]
    cluster_names = {}
    pieces = []
    position = 0
    for mention in mentions:
        start, end = int(mention["start"]), int(mention["end"])
        if start < position:
            continue
        pieces.append(html.escape(text[position:start]))
        key = (start, end)
        cluster = owners.get(key, f"unassigned:{start}")
        cluster_names.setdefault(cluster, f"E{len(cluster_names) + 1}")
        color = COLORS[(int(cluster_names[cluster][1:]) - 1) % len(COLORS)]
        pieces.append(
            f'<mark title="{html.escape(cluster_names[cluster])}" '
            f'style="background:{color}">{html.escape(text[start:end])}</mark>'
        )
        position = end
    pieces.append(html.escape(text[position:]))
    return "".join(pieces)


def _outcome(prediction: int | None, target: int) -> str:
    if prediction is None:
        return "not saved"
    result = "link" if prediction else "separate"
    expected = "link" if target else "separate"
    style = "good" if prediction == target else "bad"
    return f'<span class="{style}">{result}</span>'


def _case_table(rows: list[dict], old_by_id: dict[str, int], report_owners: dict,
                pair_threshold: float, natural: bool = False) -> str:
    rendered = []
    for row in rows:
        target = int(row["target_linked"])
        a, b = row["mention_a"], row["mention_b"]
        exact = int(normalize_text(a.get("surface")) == normalize_text(b.get("surface")))
        score = float(row["hybrid_score"])
        hybrid = int(row.get("predicted_linked", score >= pair_threshold))
        if natural:
            report = int(normalize_text(b.get("surface")) in normalize_text(a.get("surface")))
            old = row.get("old_prediction")
        else:
            report = _report_pair_prediction(report_owners[str(row["doc_id"])], row)
            old = old_by_id.get(str(row.get("pair_id")))
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('doc_id','')))}</td>"
            f"<td>{html.escape(str(row.get('category') or row.get('difficulty') or ''))}</td>"
            f"<td>{html.escape(str(a.get('surface') or ''))}<br><span class='muted'>vs.</span> "
            f"{html.escape(str(b.get('surface') or ''))}</td>"
            f"<td>{_outcome(exact, target)}</td><td>{_outcome(report, target)}</td>"
            f"<td>{_outcome(old, target)}</td><td>{_outcome(hybrid, target)} "
            f"<span class='muted'>({score:.2f})</span></td>"
            f"<td>{'linked' if target else 'different people'}<br><span class='muted'>weak map label</span></td>"
            "</tr>"
        )
    return "".join(rendered)


def _natural_rows(reference_dir: Path, scorer: HybridPairScorer) -> list[dict]:
    saved = read_json(reference_dir / "pairwise_model/test_evaluation.json")
    examples = saved["examples"]
    selected_ids = [
        "214832:76bd9a3f",  # resolved below by document/difficulty if IDs change
    ]
    source_rows = []
    for group, prediction in (
        ("false_positives_most_confident", 1),
        ("false_negatives_most_confident", 0),
        ("true_positives_sample", 1),
        ("true_negatives_sample", 0),
    ):
        for item in examples.get(group, []):
            if str(item.get("doc_id")) not in {"214832", "715969", "559713"}:
                continue
            a, b = dict(item["mention_a"]), dict(item["mention_b"])
            a.setdefault("label", "PER")
            b.setdefault("label", "PER")
            row = {"doc_id": str(item["doc_id"]), "pair_id": item.get("pair_id"),
                   "difficulty": item.get("difficulty"), "target_linked": int(item["target_linked"]),
                   "mention_a": a, "mention_b": b, "old_prediction": prediction}
            row["hybrid_score"] = scorer.score(row)
            row["predicted_linked"] = scorer.predict(row)
            source_rows.append(row)
    # Prefer the known false positive 214832, the full-name/short-name miss,
    # and one exact-surface false positive, with both outcomes represented.
    chosen = []
    for doc_id in ("214832", "715969", "559713"):
        options = [row for row in source_rows if row["doc_id"] == doc_id]
        if options:
            chosen.append(options[0])
    return chosen


def build(args: argparse.Namespace) -> dict:
    comparison = read_json(args.run_dir / "comparison_report.json")
    references = args.reference_dir
    validation_rows = _read_jsonl(args.validation_pairs) if args.validation_pairs else []
    validation_category_counts = defaultdict(int)
    for row in validation_rows:
        validation_category_counts[str(row.get("difficulty") or "unknown")] += 1
    pair_exact = read_json(references / "pairwise_model/exact_surface_baseline_test_independent_challenge.json")
    pair_report = read_json(references / "pairwise_model/report_substring_baseline_test_independent_challenge.json")
    pair_old = read_json(references / "pairwise_model/test_independent_challenge.json")
    doc_exact = read_json(references / "document_level/exact_surface_baseline_test_independent_challenge.json")
    doc_report = read_json(references / "document_level/report_substring_baseline_test_independent_challenge.json")
    doc_old = read_json(references / "document_level/cluster_test_independent_challenge_natural_threshold.json")

    old_predictions = {}
    for group, prediction in (
        ("false_positives_most_confident", 1),
        ("false_negatives_most_confident", 0),
        ("true_positives_sample", 1),
        ("true_negatives_sample", 0),
    ):
        for item in pair_old["examples"].get(group, []):
            old_predictions[str(item["pair_id"])] = prediction

    train_root = args.data_dir
    document_path = train_root / "documents/test_independent_challenge.jsonl"
    map_path = train_root / "maps/test_independent_challenge.jsonl"
    pair_path = train_root / "pairs/test_independent_challenge.jsonl"
    # The comparison bundle is compact and already contains these three files.
    if not document_path.exists():
        document_path = args.reference_dir / "challenge_dataset/documents/test_independent_challenge.jsonl"
        map_path = args.reference_dir / "challenge_dataset/maps/test_independent_challenge.jsonl"
        pair_path = args.reference_dir / "challenge_dataset/pairs/test_independent_challenge.jsonl"
    challenge_pairs = _read_jsonl(pair_path)
    challenge_predictions = {str(item["pair_id"]): item
                            for item in comparison["test"]["pairwise"].get("predictions", [])}
    # The training report stores the complete pair output under its top-level key.
    challenge_predictions = {str(item["pair_id"]): item
                             for item in comparison.get("test_pair_predictions", [])}
    _, _, report_owner_by_doc = evaluate_report_baseline(document_path, map_path, pair_path)
    report_owners = {
        doc_id: {
            key: cluster for key, cluster in owner.items()
        }
        for doc_id, owner in report_owner_by_doc.items()
    }

    rendered_pair_rows = []
    pair_row_by_id = {str(row["pair_id"]): row for row in challenge_pairs}
    for pair_id, prediction in challenge_predictions.items():
        row = pair_row_by_id[pair_id]
        row = dict(row)
        row["hybrid_score"] = float(prediction["score"])
        row["predicted_linked"] = int(prediction["predicted_linked"])
        rendered_pair_rows.append(row)
    sampled = []
    for row in rendered_pair_rows:
        pair_id = str(row["pair_id"])
        if pair_id not in old_predictions:
            continue
        target = int(row["target_linked"])
        exact = int(normalize_text(row["mention_a"].get("surface")) == normalize_text(row["mention_b"].get("surface")))
        report = _report_pair_prediction(report_owners[str(row["doc_id"])], row)
        hybrid = int(row["predicted_linked"])
        old = old_predictions[pair_id]
        row["old_correct"] = old == target
        row["hybrid_correct"] = hybrid == target
        row["base_correct"] = exact == target or report == target
        row["pair_category"] = row.get("difficulty")
        sampled.append(row)
    successes = [row for row in sampled
                 if row["hybrid_correct"] and (not row["old_correct"] or not row["base_correct"])]
    successes.sort(key=lambda row: (row["pair_category"], str(row["pair_id"])))
    failures = [row for row in sampled
                if not row["hybrid_correct"] and (row["old_correct"] or row["base_correct"])]
    failures.sort(key=lambda row: (row["pair_category"], str(row["pair_id"])))

    category_order = (
        "same_person_full_short_positive", "same_person_same_surface_positive",
        "different_people_same_surface_negative", "different_people_different_surface_negative",
    )

    def diverse_take(candidates: list[dict], limit: int, already: set[tuple]) -> list[dict]:
        chosen = []
        for category in category_order:
            options = [row for row in candidates if row["pair_category"] == category]
            if options:
                row = options[0]
                signature = (str(row["doc_id"]), category,
                             normalize_text(row["mention_a"].get("surface")),
                             normalize_text(row["mention_b"].get("surface")),
                             int(row["target_linked"]))
                if signature not in already:
                    chosen.append(row)
                    already.add(signature)
        for row in candidates:
            if len(chosen) >= limit:
                break
            category = row["pair_category"]
            signature = (str(row["doc_id"]), category,
                         normalize_text(row["mention_a"].get("surface")),
                         normalize_text(row["mention_b"].get("surface")),
                         int(row["target_linked"]))
            if signature not in already:
                chosen.append(row)
                already.add(signature)
        return chosen[:limit]

    selected_signatures: set[tuple] = set()
    selected_cases = diverse_take(successes, 4, selected_signatures)
    selected_cases.extend(diverse_take(failures, 4, selected_signatures))
    if len(selected_cases) < 6:
        selected_cases.extend(diverse_take(sampled, 6 - len(selected_cases), selected_signatures))

    scorer = HybridPairScorer.from_json(args.run_dir / "hybrid_pair_scorer.json")
    natural_cases = _natural_rows(references, scorer)

    document_records = _document_rows(document_path, map_path)
    exact_owners = {item["doc_id"]: _exact_owner(item) for item in document_records}
    hybrid_docs = {str(item["doc_id"]): item for item in comparison.get("test_document_predictions", [])}
    error_docs = defaultdict(list)
    for row in rendered_pair_rows:
        if int(row["target_linked"]) != int(float(row["hybrid_score"]) >= comparison["pair_threshold"]):
            error_docs[str(row["doc_id"])].append(row)
    visual_doc_id = next(iter(sorted(error_docs, key=lambda key: (-len(error_docs[key]), key))), None)
    if visual_doc_id is None:
        visual_doc_id = next(iter(hybrid_docs), None)
    document = next(item for item in document_records if item["doc_id"] == visual_doc_id)
    hybrid_doc = hybrid_docs[visual_doc_id]
    hybrid_owners = {
        (int(item["start"]), int(item["end"])): int(item["cluster"])
        for item in hybrid_doc["mentions"]
    }
    report_owners_for_visual = report_owners[visual_doc_id]
    document_panels = "".join(
        f"<section class='doc-panel'><h3>{title}</h3><div class='doc-text'>{_render_document(document, owners)}</div></section>"
        for title, owners in (
            ("Exact surface baseline", exact_owners[visual_doc_id]),
            ("VDT Algorithm 1 baseline", report_owners_for_visual),
            ("New hybrid", hybrid_owners),
        )
    )

    pair_methods = [
        ("Exact surface", _metric_pair(pair_exact["metrics"])),
        ("VDT Algorithm 1", _metric_pair(pair_report["metrics"])),
        ("Old transformer", _metric_pair(pair_old["metrics"])),
        ("New hybrid", comparison["test"]["pairwise"]["overall"]),
    ]
    exact_doc_metrics = {
        "documents": doc_exact["documents"],
        "b_cubed": doc_exact["b_cubed"],
        "pairwise": doc_exact["pairwise"],
        "exact_partition_documents": "not saved",
    }
    report_doc_metrics = {
        **doc_report["documents"],
        "exact_partition_documents": doc_report.get("exact_partition_documents", "not saved"),
    }
    doc_methods = [
        ("Exact surface", exact_doc_metrics),
        ("VDT Algorithm 1", report_doc_metrics),
        ("Old transformer", doc_old["metrics"]),
        ("New hybrid", comparison["test"]["document_clustering"]),
    ]
    pair_metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{metric['accuracy']:.3f}</td>"
        f"<td>{metric['precision']:.3f}</td><td>{metric['recall']:.3f}</td>"
        f"<td>{metric['f1']:.3f}</td><td>{metric.get('pairs', metric.get('examples','?'))}</td></tr>"
        for name, metric in pair_methods
    )
    doc_metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{metric['b_cubed']['precision']:.3f}</td>"
        f"<td>{metric['b_cubed']['recall']:.3f}</td><td>{metric['b_cubed']['f1']:.3f}</td>"
        f"<td>{metric['pairwise']['precision']:.3f}</td><td>{metric['pairwise']['recall']:.3f}</td>"
        f"<td>{metric['pairwise']['f1']:.3f}</td>"
        f"<td>{metric.get('exact_partition_documents', 'not saved')}/"
        f"{metric.get('documents', '?')}</td></tr>"
        for name, metric in doc_methods
    )
    example_rows = _case_table(selected_cases, old_predictions, report_owners,
                               comparison["pair_threshold"])
    natural_table = _case_table(natural_cases, {}, {}, comparison["pair_threshold"], natural=True)
    pair_categories = [
        ("same_person_full_short_positive", "Same person: full name and short name"),
        ("same_person_same_surface_positive", "Same person: repeated short name"),
        ("different_people_same_surface_negative", "Different people: same short name"),
        ("different_people_different_surface_negative", "Different people: overlapping full/short names"),
    ]
    category_table_rows = []
    new_by_category = comparison["test"]["pairwise"]["by_challenge_category"]
    exact_by_category = pair_exact["metrics"]["by_difficulty"]
    report_by_category = pair_report["metrics"]["by_difficulty"]
    old_by_category = pair_old["metrics"]["by_difficulty"]
    for key, title in pair_categories:
        values = [
            exact_by_category[key], report_by_category[key], old_by_category[key], new_by_category[key]
        ]
        category_table_rows.append(
            f"<tr><th>{html.escape(title)}</th>" + "".join(
                f"<td>{item['accuracy']:.3f} ({item.get('examples', item.get('pairs','?'))})</td>"
                for item in values
            ) + "</tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VDT Hybrid Linker Comparison</title>
<style>
body{{font:16px/1.5 system-ui,Segoe UI,sans-serif;color:#17212b;background:#f5f7fa;margin:0;padding:24px}}
main{{max-width:1500px;margin:auto}}h1,h2,h3{{line-height:1.2}}section,.notice{{background:white;border:1px solid #d9e0e8;border-radius:10px;padding:20px;margin:18px 0}}
.notice{{background:#fff8df;border-color:#e6d388}}table{{width:100%;border-collapse:collapse;margin:12px 0 22px;font-size:14px}}
th,td{{border-bottom:1px solid #e4e8ed;text-align:left;padding:9px;vertical-align:top}}th{{background:#f0f3f7}}
.good{{color:#087444;font-weight:700}}.bad{{color:#b12727;font-weight:700}}.muted{{color:#66727f;font-size:12px}}
.doc-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.doc-panel{{margin:0;padding:12px;min-width:0}}
.doc-text{{white-space:pre-wrap;word-break:break-word;max-height:650px;overflow:auto;font:13px/1.55 ui-monospace,Consolas,monospace}}
mark{{padding:1px 2px;border-radius:3px}}.small{{font-size:13px;color:#55616e}}
@media(max-width:900px){{.doc-grid{{grid-template-columns:1fr}}body{{padding:12px}}}}
</style></head><body><main>
<h1>Rule constrained hybrid linker: result comparison</h1>
<p>Pairwise training used {comparison['training']['train_pairs']:,} balanced training pairs. Thresholds were selected on {comparison['training']['validation_pairs']:,} validation pairs{f" ({', '.join(f'{count} {name}' for name, count in sorted(validation_category_counts.items()))})" if validation_rows else ''}. Test: {comparison['test']['pairwise']['overall']['pairs']:,} hard challenge pairs across {comparison['test']['document_clustering']['documents']:,} documents.</p>
<div class="notice"><strong>Read this as a reconstruction experiment.</strong> Labels are weak identities from reconstruction maps, not human verified original identities. The challenge test includes constructed alias/collision cases. The old transformer row is from saved Kaggle evaluation output; its checkpoint and per-document cluster assignments were not saved locally. The document view therefore compares the exact baseline, report baseline, and rerun hybrid only.</div>
<section><h2>Pair linking</h2><p class="small">A pair is one decision about whether two mentions refer to the same entity. Higher precision means fewer false merges; higher recall means fewer missed links.</p>
<table><thead><tr><th>System</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Pairs</th></tr></thead><tbody>{pair_metric_rows}</tbody></table>
<h3>Accuracy on each constructed challenge type</h3><table><thead><tr><th>Challenge</th><th>Exact surface</th><th>VDT Algorithm 1</th><th>Old transformer</th><th>New hybrid</th></tr></thead><tbody>{''.join(category_table_rows)}</tbody></table></section>
<section><h2>Whole-document clustering</h2><p class="small">B³ and pairwise scores compare the predicted mention groups with reconstruction-map identities. “Exact partitions” counts documents whose full clustering matched the weak map exactly.</p>
<table><thead><tr><th>System</th><th>B³ P</th><th>B³ R</th><th>B³ F1</th><th>Pair P</th><th>Pair R</th><th>Pair F1</th><th>Exact partitions</th></tr></thead><tbody>{doc_metric_rows}</tbody></table></section>
<section><h2>Saved challenge examples: where the new method wins and loses</h2><p class="small">Examples are selected from pairs for which the old transformer saved an individual prediction. The expected answer is still the reconstruction map label.</p>
<table><thead><tr><th>Document</th><th>Type</th><th>Mention pair</th><th>Exact surface</th><th>VDT Algorithm 1</th><th>Old transformer</th><th>New hybrid (decision / score)</th><th>Expected</th></tr></thead><tbody>{example_rows}</tbody></table></section>
<section><h2>Natural held-out error examples</h2><p class="small">These are saved error/sample pairs from the unaugmented reconstructed test set, not the generated collision challenge. For the report baseline, this table shows the direct ordered substring implication for the pair; the full report algorithm is summarized above at document level.</p>
<table><thead><tr><th>Document</th><th>Type</th><th>Mention pair</th><th>Exact</th><th>Substring signal</th><th>Old transformer</th><th>New hybrid (score)</th><th>Expected</th></tr></thead><tbody>{natural_table}</tbody></table></section>
<section><h2>One held-out challenge document</h2><p class="small">Document {html.escape(str(visual_doc_id))}; synthetic reconstructed challenge text. The same text appears in each column; highlight colors show which mentions each system grouped together. The old model’s per-document assignments were not saved.</p>
<div class="doc-grid">{document_panels}</div></section>
<section><h2>What the new implementation does</h2><ol><li>Reject links across different NER types, apply the existing high-confidence location rules, and block clear conflicting gendered-title cues such as “anh” versus “chị”.</li><li>Score remaining pairs with a logistic model over name overlap, local context, nearby title, procedural role, age, and existing pair features.</li><li>Choose pair and document thresholds on validation, then use complete-link clustering to avoid chaining a cluster through one weak pair.</li><li>Keep uncertain cases available for review in the surrounding application; this experiment reports binary decisions at the selected thresholds.</li></ol><p class="small">The score shown in examples is the logistic score; the displayed final decision also applies the type, location, and title rules.</p>
<p class="small">Scorer: {html.escape(str(comparison['model']))}. Pair threshold {comparison['pair_threshold']:.3f}; document threshold {comparison['cluster_threshold']:.2f}. No transformer checkpoint or GPU was used for this new run.</p></section>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    return {"html": str(args.output), "selected_challenge_examples": len(selected_cases),
            "natural_saved_examples": len(natural_cases), "visual_document": visual_doc_id,
            "validation_pairs": len(validation_rows),
            "validation_category_counts": dict(validation_category_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("run_dir", "reference_dir", "data_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.validation_pairs:
        args.validation_pairs = args.validation_pairs.resolve()
    args.output = args.output.resolve()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
