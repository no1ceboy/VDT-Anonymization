"""Build a self-contained HTML comparison from saved linker evaluation artifacts.

Usage from the repository root:
python -m src.vdt_anonymization.review.challenge_comparison --run-dir PATH --output PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..entity_linking.raw_dataset import _target_replacements
from ..entity_linking.report_baseline import report_person_roots


PAIR_CATEGORIES = [
    ("same_person_full_short_positive", "Same person, full name ↔ short name", "link"),
    ("same_person_same_surface_positive", "Same person, repeated short name", "link"),
    ("different_people_same_surface_negative", "Different people, identical short name", "separate"),
    ("different_people_different_surface_negative", "Different people, different names", "separate"),
]

MARKER_CATEGORIES = [
    ("full_name_marker", "Full-name marker"),
    ("numbered_marker", "Numbered marker"),
    ("dotted_initial", "Dotted initial"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def correct(metric: dict) -> int:
    confusion = metric["confusion"]
    return confusion["tp"] + confusion["tn"]


def find_document(path: Path, doc_id: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("doc_name") or row.get("doc_id")) == doc_id:
                    return row
    raise ValueError(f"Document {doc_id} was not found in {path}")


def find_pair(path: Path, pair_id: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("pair_id")) == pair_id:
                    return row
    raise ValueError(f"Pair {pair_id} was not found in {path}")


def report_person_owners(run_dir: Path, doc_id: str) -> dict[tuple[int, int], int]:
    document = find_document(run_dir / "challenge_dataset/documents/test_independent_challenge.jsonl", doc_id)
    mapping = find_document(run_dir / "challenge_dataset/maps/test_independent_challenge.jsonl", doc_id)
    text = str(document.get("entity_linking_input") or document.get("synthetic_markdown") or "")
    translated = _target_replacements(
        str(document.get("original_anonymized_markdown") or ""), text, mapping
    )
    entities = {str(item["entity_id"]): item for item in mapping.get("entities", [])}
    mentions = []
    for item in translated:
        mention = dict(item)
        mention["surface"] = text[int(item["start"]):int(item["end"])]
        mention["label"] = entities[str(item["entity_id"])]["label"]
        mentions.append(mention)
    mentions.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    roots = report_person_roots(mentions)
    return {
        (int(mentions[index]["start"]), int(mentions[index]["end"])): root
        for index, root in roots.items()
    }


def build_data(run_dir: Path) -> dict:
    pair_exact_baseline = read_json(run_dir / "pairwise_model/exact_surface_baseline_test_independent_challenge.json")
    pair_report_baseline = read_json(run_dir / "pairwise_model/report_substring_baseline_test_independent_challenge.json")
    pair_model = read_json(run_dir / "pairwise_model/test_independent_challenge.json")
    doc_exact_baseline = read_json(run_dir / "document_level/exact_surface_baseline_test_independent_challenge.json")
    doc_report_baseline = read_json(run_dir / "document_level/report_substring_baseline_test_independent_challenge.json")
    doc_model_default = read_json(run_dir / "document_level/cluster_test_independent_challenge_natural_threshold.json")
    doc_model_challenge = read_json(run_dir / "document_level/cluster_test_independent_challenge_challenge_threshold.json")
    manifest = read_json(run_dir / "dataset_manifest.json")
    example = read_json(run_dir / "document_level/test_challenge_example.json")
    source = find_document(run_dir / "challenge_dataset/documents/test_independent_challenge.jsonl", str(example["doc_id"]))
    text = source["entity_linking_input"]
    person_owners = report_person_owners(run_dir, str(example["doc_id"]))
    report_cluster_names = {}
    for mention in example["mentions_in_document_order"]:
        start = int(mention["position"])
        end = start + len(mention["surface"])
        if (start, end) in person_owners:
            root = person_owners[(start, end)]
            report_cluster_names.setdefault(root, f"R{len(report_cluster_names) + 1}")
            mention["report_baseline_cluster"] = report_cluster_names[root]
        else:
            mention["report_baseline_cluster"] = mention["baseline_cluster"]
    previous_end = 0
    for mention in example["mentions_in_document_order"]:
        start = mention["position"]
        end = start + len(mention["surface"])
        if start < previous_end or text[start:end] != mention["surface"]:
            raise ValueError(f"Example mention {mention['order']} does not align with reconstructed document")
        previous_end = end

    failure = next(
        item for item in pair_model["examples"]["false_positives_most_confident"]
        if str(item["doc_id"]) == "498641"
        and item["difficulty"] == "different_people_different_surface_negative"
    )
    failure_pair = find_pair(
        run_dir / "challenge_dataset/pairs/test_independent_challenge.jsonl",
        str(failure["pair_id"]),
    )
    failure_owner = report_person_owners(run_dir, str(failure["doc_id"]))
    failure_left = failure_pair["mention_a"]
    failure_right = failure_pair["mention_b"]
    failure["report_baseline_linked"] = int(
        failure_owner[(int(failure_left["start"]), int(failure_left["end"]))]
        == failure_owner[(int(failure_right["start"]), int(failure_right["end"]))]
    )
    failure["exact_baseline_linked"] = int(
        str(failure_left["surface"]).casefold().strip()
        == str(failure_right["surface"]).casefold().strip()
    )
    for pair in example["challenge_pair_decisions"]:
        left_candidates = [
            item for item in example["mentions_in_document_order"]
            if item["surface"] == pair["surface_a"] and item["identity"] in {"A", "B"}
        ]
        right_candidates = [
            item for item in example["mentions_in_document_order"]
            if item["surface"] == pair["surface_b"] and item["identity"] in {"A", "B"}
        ]
        decisions = {
            int(left["report_baseline_cluster"] == right["report_baseline_cluster"])
            for left in left_candidates for right in right_candidates
            if left is not right
            and ((left["identity"] == right["identity"]) == bool(pair["target_linked"]))
        }
        if len(decisions) != 1:
            raise ValueError(f"Saved example pair does not have one unambiguous report-baseline outcome: {pair}; outcomes={decisions}")
        pair["report_baseline_linked"] = int(next(iter(decisions)))
    exact_difficulty = pair_exact_baseline["metrics"]["by_difficulty"]
    report_difficulty = pair_report_baseline["metrics"]["by_difficulty"]
    model_difficulty = pair_model["metrics"]["by_difficulty"]
    pairs = []
    for key, title, target in PAIR_CATEGORIES:
        exact = exact_difficulty[key]
        report = report_difficulty[key]
        model = model_difficulty[key]
        pairs.append({
            "key": key,
            "title": title,
            "target": target,
            "count": model["examples"],
            "exact_baseline_correct": correct(exact),
            "report_baseline_correct": correct(report),
            "model_correct": correct(model),
            "example": next(
                item for item in example["challenge_pair_decisions"]
                if item["category"] == key.removesuffix("_positive").removesuffix("_negative")
            ),
        })
    markers = []
    for key, title in MARKER_CATEGORIES:
        exact = pair_exact_baseline["metrics"]["by_primary_challenge"][key]
        report = pair_report_baseline["metrics"]["by_primary_challenge"][key]
        model = pair_model["metrics"]["by_challenge"][key]
        markers.append({
            "title": title,
            "count": model["examples"],
            "exact_baseline_correct": correct(exact),
            "report_baseline_correct": correct(report),
            "model_correct": correct(model),
        })

    test_meta = manifest["augmentation"]["test_challenge"]
    return {
        "pairs": pairs,
        "markers": markers,
        "pair_overall": {
            "count": pair_model["metrics"]["overall"]["examples"],
            "exact_baseline_correct": correct(pair_exact_baseline["metrics"]["overall"]),
            "report_baseline_correct": correct(pair_report_baseline["metrics"]["overall"]),
            "model_correct": correct(pair_model["metrics"]["overall"]),
        },
        "documents": {
            "exact_baseline": doc_exact_baseline,
            "report_baseline": doc_report_baseline,
            "model_default": doc_model_default["metrics"],
            "model_challenge": doc_model_challenge["metrics"],
            "default_threshold": doc_model_default["threshold"],
            "challenge_threshold": doc_model_challenge["threshold"],
            "alias_only": test_meta["same_person_alias_only"],
            "collision": test_meta["same_surface_collision_with_alias_coreference"],
        },
        "example": {
            "doc_id": example["doc_id"],
            "threshold": example["threshold"],
            "text": text,
            "mentions": example["mentions_in_document_order"],
            "challenge_pairs": example["challenge_pair_decisions"],
        },
        "failure": {
            "doc_id": failure["doc_id"],
            "surface_a": failure["mention_a"]["surface"],
            "surface_b": failure["mention_b"]["surface"],
            "score": failure["predicted_score"],
            "difficulty": failure["difficulty"],
            "exact_baseline_linked": failure["exact_baseline_linked"],
            "report_baseline_linked": failure["report_baseline_linked"],
        },
    }


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VDT · Report baseline vs entity linker</title>
<style>
  :root{font-family:Inter,ui-sans-serif,system-ui,"Segoe UI",sans-serif;color:#172033;background:#f4f6fb}
  *{box-sizing:border-box}body{margin:0}button{font:inherit;cursor:pointer}main{max-width:1160px;margin:auto;padding:30px 22px 70px}
  header{background:#151f38;color:#fff;padding:42px max(22px,calc((100vw - 1116px)/2));border-bottom:5px solid #76e4c1}
  header .eyebrow{color:#8be9cb;font-size:12px;font-weight:800;letter-spacing:.15em;text-transform:uppercase}
  h1{font-size:clamp(31px,5vw,52px);line-height:1.08;margin:12px 0}h2{font-size:25px;margin:0 0 8px}h3{font-size:17px;margin:0 0 8px}
  header p{max-width:730px;color:#d6deeb;line-height:1.55;margin:0}p{line-height:1.55}.muted{color:#637087}.small{font-size:13px}
  .section{margin-top:32px}.intro{max-width:760px;margin:0 0 18px;color:#536078}
  .top-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:white;border:1px solid #e0e6f1;border-radius:16px;box-shadow:0 5px 18px #14223c0b;padding:20px}
  .metric{font-size:30px;font-weight:800;letter-spacing:-.04em}.exact{color:#65748b}.baseline{color:#d06a37}.model{color:#087c69}.label{font-size:12px;text-transform:uppercase;letter-spacing:.1em;font-weight:800;color:#637087}
  .pair-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.pair-card .head{display:flex;justify-content:space-between;gap:10px}.pill{border-radius:99px;padding:5px 10px;font-size:12px;font-weight:750;background:#eef1f8;color:#4b5a75;white-space:nowrap}
  .bar-row{display:grid;grid-template-columns:100px 1fr 76px;gap:10px;align-items:center;margin-top:13px;font-size:13px;font-weight:700}.track{height:13px;background:#edf0f7;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px}.fill.exact{background:#9caac0}.fill.baseline{background:#eaa16b}.fill.model{background:#24b39a}.value{text-align:right;font-variant-numeric:tabular-nums}
  .example{margin-top:18px;background:#f7f9fc;border:1px solid #e7ebf4;border-radius:10px;padding:12px;font-size:13px}.example .surfaces{font-weight:800;font-size:15px;margin:5px 0 9px}.decision{display:flex;gap:8px;flex-wrap:wrap}.tag{display:inline-block;border-radius:7px;padding:5px 8px;font-size:12px;font-weight:750}.good{background:#e1f7ed;color:#0a7354}.bad{background:#fff0e6;color:#a44722}
  .doc-metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.metric-table{width:100%;border-collapse:collapse;margin-top:8px}.metric-table td,.metric-table th{padding:10px 5px;text-align:left;border-bottom:1px solid #edf0f5;font-size:13px}.metric-table td:not(:first-child),.metric-table th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}.metric-table th{color:#64718a;font-weight:700}.threshold{display:flex;gap:7px;flex-wrap:wrap;margin:15px 0 0}.threshold button,.tabs button{border:1px solid #d8e0ec;background:white;border-radius:9px;padding:8px 12px;color:#4d5b72;font-weight:700}.threshold button.active,.tabs button.active{border-color:#1e7d71;background:#e3f7ef;color:#106a5c}
  .doc-panel{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(285px,.75fr);gap:14px}.tabs{display:flex;gap:7px;margin:15px 0}.doc-text{white-space:pre-wrap;overflow:auto;max-height:565px;padding:19px;background:#fcfcff;border:1px solid #e2e8f2;border-radius:12px;line-height:1.8;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px}.mention{padding:2px 3px;border-radius:5px;cursor:pointer;font-weight:750;border-bottom:2px solid transparent}.mention:hover,.mention.selected{outline:2px solid #162e63}.mention.dim{opacity:.25}.group{border:1px solid #e2e8f2;border-radius:11px;padding:12px;margin-bottom:9px;cursor:pointer;background:#fff}.group:hover,.group.selected{border-color:#1f897a;background:#f2fffa}.group strong{display:block;margin-bottom:3px}.group small{color:#5d6b80}.group .names{font-size:13px;line-height:1.4;margin-top:5px}.note{padding:13px 15px;border-left:4px solid #87a5cd;background:#edf3fb;color:#42516c;border-radius:0 9px 9px 0;font-size:13px;line-height:1.5}
  .failure{border-left:5px solid #db754c}.flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:12px}.flow span{background:#eef4fc;border:1px solid #d8e5f6;border-radius:9px;padding:9px 12px;font-size:13px;font-weight:750}.flow b{color:#76849b}
  @media(max-width:760px){.top-grid,.pair-grid,.doc-metrics,.doc-panel{grid-template-columns:1fr}header{padding:32px 22px}.metric{font-size:27px}}
</style></head><body>
<header><div class="eyebrow">VDT · reconstructed court text</div><h1>Report baseline vs entity linker</h1><p>Comparison with the ordered substring rule described in the VDT phase 2 report. The exact-match baseline from the earlier comparison is shown separately. All labels here come from reconstruction maps, not independent human annotation.</p></header>
<main>
  <section class="top-grid" id="summary"></section>
  <section class="section"><h2>All four hard pair situations</h2><p class="intro">Held-out challenge sample: 180 pairs of each type, using the saved pair-decision threshold of 0.80. Bars show correct decisions out of 180. A negative-only row has no useful positive-class F1, so this view shows the number classified correctly.</p><div class="pair-grid" id="pairs"></div></section>
  <section class="section"><h2>Original marker context</h2><p class="intro">The same 720 pairs, regrouped by the source document's marker form. These rows overlap the four situations above; do not add their totals to 720.</p><div class="top-grid" id="markers"></div></section>
  <section class="section"><h2>Whole-document results</h2><p class="intro">230 challenge documents: 103 with two people sharing a short name, and 127 with a same-person alias change only. B³ F1 measures mention-level cluster agreement; cluster-pair F1 checks whether each pair ends up in the same cluster.</p><div class="doc-metrics"><div class="card"><h3>Cluster agreement</h3><div class="threshold"><button data-threshold="default" class="active">General threshold 0.10</button><button data-threshold="challenge">Challenge threshold 0.05</button></div><table class="metric-table"><thead><tr><th>Metric</th><th>Exact match</th><th>VDT report rule</th><th>Entity linker</th></tr></thead><tbody id="doc-metrics"></tbody></table><p class="small muted" id="threshold-note"></p></div><div class="card"><h3>What changes</h3><p>The report baseline maps a later person mention to an earlier one when the later text is a substring of the earlier text. Location handling stays on the existing rule-based path. The linker scores mention pairs from context, then complete-link clustering forms document identities.</p><div class="flow"><span>Document text</span><b>→</b><span>NER mentions</span><b>→</b><span>pair scores</span><b>→</b><span>clusters</span></div><p class="small muted">The pair table above contains a balanced subset. These document metrics use every evaluated mention in the 230 challenge documents. They remain scores against weak reconstruction labels.</p></div></div></section>
  <section class="section"><h2>Explore a real saved cluster example</h2><p class="intro">Document <strong id="example-id"></strong> contains two reconstructed people called “Ha.” Click a mention or cluster. The colors show each system's groups; A/B identity tags are weak labels and are not given to either system.</p><div class="tabs"><button data-view="baseline">Exact-match groups</button><button data-view="report_baseline" class="active">VDT report groups</button><button data-view="linker">Entity linker groups</button></div><div class="doc-panel"><div class="card"><h3>Reconstructed document text</h3><div class="doc-text" id="doc-text"></div></div><div class="card"><h3 id="cluster-title">Groups</h3><div id="groups"></div><div class="note">The example uses threshold <strong id="example-threshold"></strong>. Its person groups are whole-document outputs; the text is reconstructed, not an original unmasked court record.</div></div></div></section>
  <section class="section"><div class="card failure"><h2>A shared-name collision</h2><p>In document <strong id="failure-id"></strong>, <strong>Lê Thị Hung</strong> and <strong>Nguyễn Thị Hung</strong> are different people. The exact-match baseline separates the full name from the short mention. The report substring rule and the linker both join <strong id="failure-pair"></strong>: the rule attaches “Hung” to the first earlier full name containing it.</p><div class="flow"><span id="failure-score"></span></div><p class="small muted">This illustrates the report baseline's collision weakness. The entity linker also makes this false merge on the weak reconstructed labels.</p></div></section>
  <section class="section"><div class="note">Interpretation: these are development results against synthetic reconstructed names and weak reconstruction-map identities. They measure the linker with supplied mentions. They do not establish accuracy of NER, anonymized output, or performance on independently labeled real documents.</div></section>
</main><script id="report-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id);
const pct=(a,b)=>(100*a/b).toFixed(1)+'%';
const f1=x=>(100*x).toFixed(1)+'%';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const resultTag=(decision,target)=>`<span class="tag ${decision===target?'good':'bad'}">${decision===target?'✓':'✕'} ${decision==='link'?'Link':'Separate'}</span>`;
const overall=data.pair_overall;
$('summary').innerHTML=`<div class="card"><div class="label">Exact-match · hard pairs</div><div class="metric exact">${pct(overall.exact_baseline_correct,overall.count)}</div><div class="muted">${overall.exact_baseline_correct} / ${overall.count} correct</div></div><div class="card"><div class="label">VDT report rule · hard pairs</div><div class="metric baseline">${pct(overall.report_baseline_correct,overall.count)}</div><div class="muted">${overall.report_baseline_correct} / ${overall.count} correct</div></div><div class="card"><div class="label">Entity linker · hard pairs</div><div class="metric model">${pct(overall.model_correct,overall.count)}</div><div class="muted">${overall.model_correct} / ${overall.count} correct</div></div>`;
function bar(label,value,total,type){return `<div class="bar-row"><span>${label}</span><div class="track"><div class="fill ${type}" style="width:${100*value/total}%"></div></div><span class="value">${value}/${total}</span></div>`}
 $('pairs').innerHTML=data.pairs.map(p=>{let e=p.example;let target=p.target;let exact=e.baseline_linked?'link':'separate';let report=e.report_baseline_linked?'link':'separate';let model=e.linker_pair_decision?'link':'separate';return `<div class="card pair-card"><div class="head"><h3>${esc(p.title)}</h3><span class="pill">Should ${target}</span></div>${bar('Exact match',p.exact_baseline_correct,p.count,'exact')}${bar('VDT report',p.report_baseline_correct,p.count,'baseline')}${bar('Linker',p.model_correct,p.count,'model')}<div class="example"><div class="label">Saved example · document ${esc(data.example.doc_id)}</div><div class="surfaces">${esc(e.surface_a)} ↔ ${esc(e.surface_b)}</div><div class="decision">Exact ${resultTag(exact,target)} &nbsp; Report ${resultTag(report,target)} &nbsp; Linker ${resultTag(model,target)} <span class="pill">score ${e.linker_score.toFixed(4)}</span></div></div></div>`}).join('');
$('markers').innerHTML=data.markers.map(m=>`<div class="card"><h3>${esc(m.title)} <span class="pill">${m.count} pairs</span></h3>${bar('Exact match',m.exact_baseline_correct,m.count,'exact')}${bar('VDT report',m.report_baseline_correct,m.count,'baseline')}${bar('Linker',m.model_correct,m.count,'model')}</div>`).join('');
function showThreshold(which){let d=data.documents,m=which==='default'?d.model_default:d.model_challenge;let t=which==='default'?d.default_threshold:d.challenge_threshold;let r=d.report_baseline.documents;$('doc-metrics').innerHTML=`<tr><td>B³ F1</td><td>${f1(d.exact_baseline.b_cubed.f1)}</td><td><strong>${f1(r.b_cubed.f1)}</strong></td><td><strong class="model">${f1(m.b_cubed.f1)}</strong></td></tr><tr><td>Cluster-pair F1</td><td>${f1(d.exact_baseline.pairwise.f1)}</td><td><strong>${f1(r.pairwise.f1)}</strong></td><td><strong class="model">${f1(m.pairwise.f1)}</strong></td></tr><tr><td>Exact document partitions</td><td>not saved</td><td>${d.report_baseline.exact_partition_documents}/${r.documents}</td><td>${m.exact_partition_documents}/${m.documents}</td></tr>`;$('threshold-note').textContent=`Linker threshold ${t.toFixed(2)}. The report-rule exact-partition count was recomputed from the same 230 documents.`;document.querySelectorAll('[data-threshold]').forEach(b=>b.classList.toggle('active',b.dataset.threshold===which))}
document.querySelectorAll('[data-threshold]').forEach(b=>b.addEventListener('click',()=>showThreshold(b.dataset.threshold)));showThreshold('default');
let view='baseline',selected=null;const m=data.example.mentions;const palette=['#b6e2ff','#ffd9a6','#c8eecf','#e4d7ff','#f6cbdc','#d4ece7','#f2e3b7','#cbd8f6','#d7e9a7','#ead6cc'];
function clusterId(mention){return view==='report_baseline'?mention.report_baseline_cluster:view==='baseline'?mention.baseline_cluster:mention.linker_cluster}
function groups(){let g=new Map();m.forEach(x=>{let id=clusterId(x);if(!g.has(id))g.set(id,[]);g.get(id).push(x)});return [...g.entries()]}
function renderDoc(){let text=data.example.text,pos=0,out='';m.forEach((x,i)=>{out+=esc(text.slice(pos,x.position));let id=clusterId(x);let all=groups(),idx=all.findIndex(([key])=>key===id);let style=`background:${palette[idx%palette.length]}`;let dim=selected&&selected!==id?' dim':'';let sel=selected===id?' selected':'';out+=`<mark class="mention${dim}${sel}" data-index="${i}" style="${style}" title="${esc(id)} · evaluation identity ${esc(x.identity)}">${esc(x.surface)}</mark>`;pos=x.position+x.surface.length});out+=esc(text.slice(pos));$('doc-text').innerHTML=out;document.querySelectorAll('.mention').forEach(el=>el.addEventListener('click',()=>{let id=clusterId(m[Number(el.dataset.index)]);selected=selected===id?null:id;renderDoc();renderGroups()}))}
function renderGroups(){let all=groups();let title=view==='baseline'?'Exact-match baseline':view==='report_baseline'?'VDT report rule':'Entity linker';$('cluster-title').textContent=title+` groups (${all.length})`;$('groups').innerHTML=all.map(([id,ms],idx)=>{let names=[...new Set(ms.map(x=>x.surface))];let identities=[...new Set(ms.map(x=>x.identity).filter(x=>x!=='other'))];let mixed=identities.includes('A')&&identities.includes('B');let identityText=identities.length?'Evaluation identity: '+identities.join(' + '):'Outside the A/B example';return `<div class="group ${selected===id?'selected':''}" data-group="${esc(id)}" style="border-left:7px solid ${palette[idx%palette.length]}"><strong>${esc(id)} · ${ms.length} mention${ms.length===1?'':'s'} ${mixed?'⚠️ mixed people':''}</strong><small>${esc(identityText)}</small><div class="names">${esc(names.join(' · '))}</div></div>`}).join('');document.querySelectorAll('.group').forEach(el=>el.addEventListener('click',()=>{selected=selected===el.dataset.group?null:el.dataset.group;renderDoc();renderGroups()}))}
function showView(next){view=next;selected=null;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));renderDoc();renderGroups()}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>showView(b.dataset.view)));$('example-id').textContent=data.example.doc_id;$('example-threshold').textContent=data.example.threshold.toFixed(2);showView('report_baseline');
$('failure-id').textContent=data.failure.doc_id;$('failure-pair').textContent=`${data.failure.surface_a} ↔ ${data.failure.surface_b}`;$('failure-score').textContent=`exact match: ${data.failure.exact_baseline_linked?'link':'separate'} · report rule: ${data.failure.report_baseline_linked?'link':'separate'} · linker score ${data.failure.score.toFixed(4)}`;
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(build_data(args.run_dir), ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
