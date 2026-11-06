"""Prepare and serve a blind, local human entity-linking review set.

The annotation UI only receives original anonymized text and candidate spans.
Weak reconstruction maps are used in-memory to find informative documents and
to suggest spans; their entity IDs, replacements, and selection strata are
never sent to the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_OUTPUT = ROOT / "outputs" / "manual_linking_original_kaggle_v1"
DEFAULT_DOCUMENTS = ROOT / "outputs" / "entity_linking_v3_location_strict_20260924" / "documents" / "test.jsonl"
DEFAULT_MAPS = ROOT / "outputs" / "entity_linking_v3_location_strict_20260924" / "maps" / "test.jsonl"
DEFAULT_CHALLENGES = ROOT / "outputs" / "v3_retry_download_20260924" / "vdt_clean_10k_v3" / "challenge_rejected.jsonl"
SCHEMA_VERSION = "manual-linking-original-v1"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_jsonl_atomic(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def normalize_surface(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _province_near(text: str, start: int, end: int) -> str | None:
    context = text[max(0, start - 260):min(len(text), end + 180)]
    matches = re.findall(
        r"\b(?:tỉnh|thành\s+phố|TP\.?|TP\.\s*HCM)\s+([^,;\n.]+)",
        context,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    value = normalize_surface(matches[-1]).strip(" :-–")
    # Avoid treating long trailing prose as part of a province name.
    return " ".join(value.split()[:4]) or None


def analyze_map(doc: dict, text: str) -> tuple[set[str], list[dict]]:
    """Return weak-label mining strata and source-aligned span suggestions."""
    replacements = doc.get("replacements") or []
    entities = {str(e.get("entity_id")): str(e.get("label", ""))
                for e in doc.get("entities", []) if e.get("entity_id")}
    by_surface = defaultdict(list)
    by_entity = defaultdict(set)
    by_person_tail = defaultdict(list)
    by_location_surface = defaultdict(list)
    valid_spans = []

    for replacement in replacements:
        try:
            start, end = int(replacement["start"]), int(replacement["end"])
        except (KeyError, TypeError, ValueError):
            continue
        surface = str(replacement.get("original", ""))
        entity_id = str(replacement.get("entity_id", ""))
        label = entities.get(entity_id, str(replacement.get("label", ""))).upper()
        if not surface or start < 0 or end <= start or end > len(text) or text[start:end] != surface:
            continue
        valid_spans.append({"start": start, "end": end, "text": surface})
        normalized = normalize_surface(surface)
        if not normalized:
            continue
        by_surface[(normalized, label)].append((entity_id, start, end, surface))
        by_entity[(entity_id, label)].add(normalized)
        tokens = normalized.split()
        if label == "PER" and len(tokens) >= 2:
            by_person_tail[tokens[-1]].append((entity_id, normalized))
        if label == "LOC":
            by_location_surface[normalized].append(
                (entity_id, _province_near(text, start, end))
            )

    strata: set[str] = set()
    for (surface, _label), items in by_surface.items():
        entity_ids = {identity for identity, *_ in items if identity}
        if len(entity_ids) > 1:
            strata.add("same_surface_multiple_entities")
    if any(len(surfaces) > 1 for (identity, _label), surfaces in by_entity.items() if identity):
        strata.add("one_entity_multiple_surfaces")
    if any(len({identity for identity, _ in items if identity}) > 1
           for items in by_person_tail.values()):
        strata.add("person_shared_final_token")
    if any(len({province for _identity, province in items if province}) > 1
           for items in by_location_surface.values()):
        strata.add("location_same_name_different_province")
    return strata, valid_spans


def select_hard_test(records: list[dict], target: int) -> list[dict]:
    remaining = [record for record in records if record["strata"]]
    selected = []
    covered: set[str] = set()
    while remaining and len(selected) < target:
        ranked = sorted(
            remaining,
            key=lambda row: (
                -len(row["strata"] - covered),
                len(row["map_spans"]),
                len(row["text"]),
                hashlib.sha1(row["doc_id"].encode()).hexdigest(),
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        covered.update(chosen["strata"])
        remaining.remove(chosen)
    return selected


def select_rejected(rows: list[dict], target: int, excluded_ids: set[str], max_chars: int) -> list[dict]:
    candidates = []
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        text = str(row.get("source_markdown", ""))
        stats = row.get("reconstruction_stats") or {}
        if not doc_id or doc_id in excluded_ids or not text or len(text) > max_chars:
            continue
        count = int(stats.get("mentions", 0) or 0)
        if count < 4 or count > 100:
            continue
        strata = {f"reason:{reason}" for reason in row.get("review_reasons", [])}
        strata.update(f"feature:{key}" for key, value in (row.get("features") or {}).items()
                      if key not in {"all_marker_tokens", "relevant"} and value)
        if not strata:
            strata.add("rejected:other")
        candidates.append({
            "doc_id": doc_id,
            "text": text,
            "category": str(row.get("category", "Unknown")),
            "instance_level": str(row.get("instance_level", "Unknown")),
            "strata": strata,
            "mention_estimate": count,
            "kind": "automation_rejected",
        })

    selected = []
    covered: set[str] = set()
    while candidates and len(selected) < target:
        candidates.sort(key=lambda row: (
            -len(row["strata"] - covered),
            row["mention_estimate"],
            len(row["text"]),
            hashlib.sha1(row["doc_id"].encode()).hexdigest(),
        ))
        chosen = candidates.pop(0)
        selected.append(chosen)
        covered.update(chosen["strata"])
    return selected


def select_controls(records: list[dict], target: int, excluded_ids: set[str], seed: int) -> list[dict]:
    eligible = [row for row in records
                if row["doc_id"] not in excluded_ids
                and len(row["text"]) <= 30000
                and 5 <= len(row["map_spans"]) <= 55]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = eligible[:target]
    return [{**row, "kind": "control", "strata": set()} for row in selected]


def _load_selection(documents_file: Path, maps_file: Path, challenge_file: Path,
                    hard_count: int, rejected_count: int, control_count: int,
                    seed: int, max_chars: int, max_rejected_chars: int):
    documents = {}
    for _line, row in iter_jsonl(documents_file):
        doc_id = str(row.get("doc_name", row.get("doc_id", "")))
        text = str(row.get("original_anonymized_markdown", ""))
        if doc_id and text:
            documents[doc_id] = {
                "doc_id": doc_id,
                "text": text,
                "category": str(row.get("category", "Unknown")),
                "instance_level": str(row.get("instance_level", "Unknown")),
            }
    records = []
    for _line, map_doc in iter_jsonl(maps_file):
        doc_id = str(map_doc.get("doc_id", ""))
        row = documents.get(doc_id)
        if not row:
            continue
        strata, map_spans = analyze_map(map_doc, row["text"])
        if len(row["text"]) <= max_chars and 3 <= len(map_spans) <= 90:
            records.append({**row, "strata": strata, "map_spans": map_spans,
                            "kind": "heldout_hard"})

    hard = select_hard_test(records, hard_count)
    selected_ids = {row["doc_id"] for row in hard}
    rejected_rows = [row for _line, row in iter_jsonl(challenge_file)]
    rejected = select_rejected(rejected_rows, rejected_count, selected_ids, max_rejected_chars)
    selected_ids.update(row["doc_id"] for row in rejected)
    controls = select_controls(records, control_count, selected_ids, seed)
    selected = hard + rejected + controls
    if len({row["doc_id"] for row in selected}) != len(selected):
        raise ValueError("Document IDs are not unique after sampling")
    return selected, {
        "available_heldout_documents": len(records),
        "available_heldout_hard_documents": sum(bool(row["strata"]) for row in records),
        "available_rejected_challenge_documents": len(rejected_rows),
    }


def prepare(args) -> None:
    for path in (args.documents, args.maps, args.challenges):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = [name for name in ("kaggle_ner_input.jsonl", "queue.jsonl", "annotations.json", "selection_manifest.json")
                if (args.output / name).exists()]
    if existing:
        annotation_file = args.output / "annotations.json"
        annotation_docs = None
        if annotation_file.is_file():
            annotation_docs = json.loads(annotation_file.read_text(encoding="utf-8")).get("documents", {})
        if not args.replace_empty or annotation_docs:
            raise FileExistsError(
                f"Refusing to replace existing review data in {args.output} ({', '.join(existing)}). "
                "Choose a new --output folder; --replace-empty is allowed only when no labels have been saved."
            )
    selected, availability = _load_selection(
        args.documents, args.maps, args.challenges,
        args.hard_docs, args.rejected_docs, args.control_docs,
        args.seed, args.max_chars, args.max_rejected_chars,
    )
    if not selected:
        raise RuntimeError("No documents matched the review sampling criteria")
    private_selection = []
    kaggle_rows = []
    for row in selected:
        # This file is the only data the Kaggle NER kernel needs. It contains
        # original masked text and opaque document IDs, never reconstructions.
        kaggle_rows.append({"doc_id": row["doc_id"], "text": row["text"]})
        private_selection.append({
            "doc_id": row["doc_id"],
            "sampling_group": row["kind"],
            "selection_strata": sorted(row.get("strata", set())),
            "source_category": row.get("category", "Unknown"),
            "source_instance_level": row.get("instance_level", "Unknown"),
            "map_spans": [{key: span[key] for key in ("start", "end", "text")}
                           for span in row.get("map_spans", [])],
        })

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output / "kaggle_ner_input.jsonl", kaggle_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "human within-document entity clustering on original anonymized source text",
        "source_split": "held-out V3 test documents plus disjoint rejected challenge pool",
        "ner_execution": "Kaggle only; no model inference is performed by the local review app",
        "sampling_policy": {
            "heldout_hard": "weak-map mining: same anonymized surface across hidden identities; one hidden identity with multiple surfaces; shared person final token; same location surface with different local province context",
            "automation_rejected": "greedy coverage across recorded reconstruction rejection reasons and challenge features; low/moderate prior mention count",
            "controls": "random held-out docs not selected into hard strata",
            "seed": args.seed,
            "max_source_chars": args.max_chars,
            "max_rejected_source_chars": args.max_rejected_chars,
        },
        "availability": availability,
        "counts": {
            "documents": len(kaggle_rows),
            "sampling_groups": {group: sum(s["sampling_group"] == group for s in private_selection)
                                for group in ("heldout_hard", "automation_rejected", "control")},
        },
        "private_selection": private_selection,
        "evaluation_warning": "Sampling is intentionally challenge-enriched. Report challenge and control results separately; do not treat this set as a population estimate. Labels mined from reconstruction maps are not human truth and were hidden from annotators.",
    }
    write_json_atomic(args.output / "selection_manifest.json", manifest)
    if not (args.output / "annotations.json").exists():
        write_json_atomic(args.output / "annotations.json", {"schema_version": SCHEMA_VERSION, "documents": {}})
    print(f"[REVIEW] Prepared {len(kaggle_rows)} original masked docs for Kaggle-only NER.", flush=True)
    print(f"[REVIEW] Input: {args.output / 'kaggle_ner_input.jsonl'}", flush=True)
    print("[REVIEW] No NER model was loaded or run locally.", flush=True)


def build_queue(args) -> None:
    input_path = args.output / "kaggle_ner_input.jsonl"
    manifest_path = args.output / "selection_manifest.json"
    for path in (input_path, manifest_path, args.predictions_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (args.output / "queue.jsonl").exists():
        raise FileExistsError(f"Queue already exists: {args.output / 'queue.jsonl'}; choose another output folder.")
    input_rows = [row for _line, row in iter_jsonl(input_path)]
    by_id = {str(row["doc_id"]): row for row in input_rows}
    if len(by_id) != len(input_rows):
        raise ValueError("Kaggle input contains duplicate document IDs")
    predictions = {}
    for _line, row in iter_jsonl(args.predictions_file):
        doc_id = str(row.get("doc_id", ""))
        if doc_id in predictions:
            raise ValueError(f"Duplicate Kaggle prediction for {doc_id}")
        predictions[doc_id] = row
    missing = set(by_id) - set(predictions)
    extra = set(predictions) - set(by_id)
    if missing or extra:
        raise ValueError(f"NER output document IDs do not match the input; missing={sorted(missing)}, extra={sorted(extra)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = {str(row["doc_id"]): row for row in manifest.get("private_selection", [])}
    queue = []
    for doc_id, source in by_id.items():
        text = str(source["text"])
        candidates = {}
        for span in selected.get(doc_id, {}).get("map_spans", []):
            start, end = int(span["start"]), int(span["end"])
            if 0 <= start < end <= len(text) and text[start:end] == span["text"]:
                candidates[(start, end)] = span["text"]
        prediction = predictions[doc_id]
        for span in prediction.get("entities", []):
            try:
                start, end = int(span["start"]), int(span["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= start < end <= len(text)):
                continue
            surface = text[start:end]
            if span.get("text") and str(span["text"]) != surface:
                continue
            if surface.strip():
                candidates.setdefault((start, end), surface)
        mentions = [
            {"id": f"M{index:04d}", "start": start, "end": end, "text": surface}
            for index, ((start, end), surface) in enumerate(sorted(candidates.items()), 1)
        ]
        queue.append({"doc_id": doc_id, "text": text, "mentions": mentions})

    write_jsonl_atomic(args.output / "queue.jsonl", queue)
    manifest["counts"].update({
        "documents_with_candidates": sum(bool(row["mentions"]) for row in queue),
        "candidate_mentions": sum(len(row["mentions"]) for row in queue),
    })
    write_json_atomic(manifest_path, manifest)
    print(f"[REVIEW] Built blind annotation queue: {len(queue)} docs, {manifest['counts']['candidate_mentions']} spans.")
    print("[REVIEW] The local step only validated and merged Kaggle offsets; it did not load a model.")


def _read_queue(output_dir: Path):
    queue_path = output_dir / "queue.jsonl"
    if not queue_path.is_file():
        raise FileNotFoundError(f"Review queue not found: {queue_path}. Run the prepare command first.")
    rows = [row for _line, row in iter_jsonl(queue_path)]
    for row in rows:
        if set(row) != {"doc_id", "text", "mentions"}:
            raise ValueError(f"Unsafe or invalid queue fields for document {row.get('doc_id')}")
        if not isinstance(row["text"], str) or not isinstance(row["mentions"], list):
            raise ValueError(f"Invalid text/mentions for document {row.get('doc_id')}")
        for mention in row["mentions"]:
            start, end = int(mention["start"]), int(mention["end"])
            if row["text"][start:end] != mention["text"]:
                raise ValueError(f"Candidate offset mismatch in {row['doc_id']}:{mention['id']}")
    return rows


def make_handler(output_dir: Path):
    queue = _read_queue(output_dir)
    by_id = {row["doc_id"]: row for row in queue}
    annotations_path = output_dir / "annotations.json"
    index = {"schema_version": SCHEMA_VERSION, "documents": {}}
    if annotations_path.is_file():
        index = json.loads(annotations_path.read_text(encoding="utf-8"))
    index.setdefault("documents", {})

    class Handler(BaseHTTPRequestHandler):
        server_version = "VDTManualReview/1.0"

        def log_message(self, format_string, *args):
            print(f"[REVIEW HTTP] {self.address_string()} {format_string % args}", flush=True)

        def _headers(self, content_type="application/json; charset=utf-8"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()

        def _json(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                # Only the redacted public queue and annotations are returned.
                self._json({"schema_version": SCHEMA_VERSION, "queue": queue,
                            "annotations": index["documents"]})
                return
            assets = {
                "/": ("manual_linking.html", "text/html; charset=utf-8"),
                "/manual_linking.css": ("manual_linking.css", "text/css; charset=utf-8"),
                "/manual_linking.js": ("manual_linking.js", "text/javascript; charset=utf-8"),
            }
            if path not in assets:
                self._json({"error": "not found"}, 404)
                return
            filename, content_type = assets[path]
            body = (Path(__file__).with_name(filename)).read_bytes()
            self._headers(content_type)
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotation":
                self._json({"error": "not found"}, 404)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length < 1 or content_length > 2_000_000:
                    raise ValueError("invalid request size")
                value = json.loads(self.rfile.read(content_length))
                doc_id = str(value.get("doc_id", ""))
                if doc_id not in by_id:
                    raise ValueError("unknown document")
                saved = validate_annotation(value, by_id[doc_id])
                index["documents"][doc_id] = saved
                write_json_atomic(annotations_path, index)
                self._json({"ok": True, "saved_at": saved["updated_at"]})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)

    return Handler


def validate_annotation(value: dict, doc: dict) -> dict:
    doc_id = str(value.get("doc_id", ""))
    allowed_decisions = {"pending", "linked", "singleton", "not_entity", "uncertain", "span_error"}
    base_mentions = {item["id"]: item for item in doc["mentions"]}
    extras = value.get("extra_mentions", [])
    if not isinstance(extras, list) or len(extras) > 2000:
        raise ValueError("invalid extra mentions")
    for extra in extras:
        mention_id = str(extra.get("id", ""))
        if not re.fullmatch(r"X\d{4,}", mention_id) or mention_id in base_mentions:
            raise ValueError("invalid custom mention ID")
        start, end = int(extra["start"]), int(extra["end"])
        if start < 0 or end <= start or end > len(doc["text"]):
            raise ValueError(f"invalid custom mention offsets: {mention_id}")
        if doc["text"][start:end] != str(extra.get("text", "")):
            raise ValueError(f"custom mention text does not match source: {mention_id}")
        base_mentions[mention_id] = {"id": mention_id, "start": start, "end": end,
                                     "text": doc["text"][start:end]}

    decisions = value.get("decisions", {})
    if not isinstance(decisions, dict) or set(decisions) - set(base_mentions):
        raise ValueError("decisions reference unknown mentions")
    clean_decisions = {}
    for mention_id, row in decisions.items():
        if not isinstance(row, dict):
            raise ValueError("invalid mention decision")
        decision = str(row.get("decision", "pending"))
        if decision not in allowed_decisions:
            raise ValueError(f"invalid decision: {decision}")
        cluster_id = row.get("cluster_id")
        if decision == "linked":
            if not isinstance(cluster_id, str) or not re.fullmatch(r"E\d{2,}", cluster_id):
                raise ValueError(f"linked mention {mention_id} needs a valid cluster")
        else:
            cluster_id = None
        clean_decisions[mention_id] = {"decision": decision, "cluster_id": cluster_id}

    note = str(value.get("note", ""))[:4000]
    status = str(value.get("status", "in_progress"))
    if status not in {"in_progress", "complete", "uncertain"}:
        raise ValueError("invalid document status")
    if status == "complete":
        unresolved = [mention_id for mention_id in base_mentions
                      if clean_decisions.get(mention_id, {}).get("decision", "pending") == "pending"]
        if unresolved:
            raise ValueError(f"{len(unresolved)} mentions still need a decision")
    return {
        "doc_id": doc_id,
        "status": status,
        "note": note,
        "decisions": clean_decisions,
        "extra_mentions": [{"id": str(row["id"]), "start": int(row["start"]),
                             "end": int(row["end"]), "text": str(row["text"])} for row in extras],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _binary_metrics(examples: list[tuple[int, int]]) -> dict:
    tp = sum(gold == pred == 1 for gold, pred in examples)
    fp = sum(gold == 0 and pred == 1 for gold, pred in examples)
    tn = sum(gold == pred == 0 for gold, pred in examples)
    fn = sum(gold == 1 and pred == 0 for gold, pred in examples)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pairs": len(examples),
        "accuracy": (tp + tn) / len(examples) if examples else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate(args) -> None:
    queue = _read_queue(args.output)
    annotations_path = args.output / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"No annotations saved yet: {annotations_path}")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8")).get("documents", {})
    manifest_path = args.output / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    groups = {row["doc_id"]: row.get("sampling_group", "unknown")
              for row in manifest.get("private_selection", [])}
    per_group = defaultdict(list)
    pair_rows = []
    docs_complete = 0
    docs_uncertain = 0
    reviewed_mentions = 0
    same_surface_hard_negatives = 0
    same_entity_surface_variants = 0
    incomplete_docs = []

    for doc in queue:
        doc_id = doc["doc_id"]
        state = annotations.get(doc_id)
        if not state:
            incomplete_docs.append(doc_id)
            continue
        if state.get("status") == "uncertain":
            docs_uncertain += 1
            continue
        if state.get("status") != "complete":
            incomplete_docs.append(doc_id)
            continue
        docs_complete += 1
        mentions = list(doc["mentions"]) + list(state.get("extra_mentions", []))
        decisions = state.get("decisions", {})
        labeled = []
        for mention in mentions:
            decision = decisions.get(mention["id"], {})
            kind = decision.get("decision", "pending")
            if kind == "linked":
                key = f"linked:{decision.get('cluster_id')}"
            elif kind == "singleton":
                key = f"singleton:{mention['id']}"
            else:
                continue
            labeled.append((mention, key))
        reviewed_mentions += len(labeled)
        group = groups.get(doc_id, "unknown")
        for left_index in range(len(labeled)):
            mention_a, entity_a = labeled[left_index]
            normalized_a = normalize_surface(mention_a["text"])
            tail_a = normalized_a.split()[-1] if normalized_a else ""
            for right_index in range(left_index + 1, len(labeled)):
                mention_b, entity_b = labeled[right_index]
                normalized_b = normalize_surface(mention_b["text"])
                tail_b = normalized_b.split()[-1] if normalized_b else ""
                gold = int(entity_a == entity_b)
                same_surface = bool(normalized_a and normalized_a == normalized_b)
                same_tail = bool(tail_a and tail_a == tail_b and len(normalized_a.split()) >= 2 and len(normalized_b.split()) >= 2)
                for rule_name, prediction in (
                    ("exact_surface", int(same_surface)),
                    ("same_final_token", int(same_tail)),
                    ("surface_or_final_token", int(same_surface or same_tail)),
                ):
                    per_group[(group, rule_name)].append((gold, prediction))
                if same_surface and not gold:
                    same_surface_hard_negatives += 1
                if gold and not same_surface:
                    same_entity_surface_variants += 1
                pair_rows.append({
                    "doc_id": doc_id,
                    "sampling_group": group,
                    "mention_a": {key: mention_a[key] for key in ("id", "start", "end", "text")},
                    "mention_b": {key: mention_b[key] for key in ("id", "start", "end", "text")},
                    "target_linked": gold,
                    "heuristics": {
                        "exact_surface": int(same_surface),
                        "same_final_token": int(same_tail),
                        "surface_or_final_token": int(same_surface or same_tail),
                    },
                })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "human-labeled within-document clustering vs transparent surface heuristics",
        "documents": {"queue": len(queue), "complete": docs_complete,
                      "uncertain": docs_uncertain, "not_complete": len(incomplete_docs)},
        "reviewed_mentions_in_complete_docs": reviewed_mentions,
        "pair_counts": {"all_scored": len(pair_rows),
                        "same_surface_but_human_distinct": same_surface_hard_negatives,
                        "same_human_entity_different_surface": same_entity_surface_variants},
        "metrics_by_sampling_group": {},
        "incomplete_doc_ids": incomplete_docs,
        "interpretation": [
            "This is a challenge-enriched pilot, not a population-prevalence estimate.",
            "Rules are evaluated on original anonymized surfaces; no generated names or weak IDs are used as labels.",
            "Encoder performance must be measured on these human labels separately; no encoder score is inferred here.",
            "Only documents marked complete contribute. Uncertain documents are kept out of the metrics.",
        ],
    }
    for (group, rule_name), examples in sorted(per_group.items()):
        summary["metrics_by_sampling_group"].setdefault(group, {})[rule_name] = _binary_metrics(examples)

    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / "human_eval.json", summary)
    write_jsonl_atomic(args.output / "human_gold_pairs.jsonl", pair_rows)
    print(f"[REVIEW] Complete docs: {docs_complete}; uncertain: {docs_uncertain}; incomplete: {len(incomplete_docs)}")
    print(f"[REVIEW] Human-labeled candidate pairs: {len(pair_rows)}")
    print(f"[REVIEW] Surface-rule hard negatives: {same_surface_hard_negatives}; same-entity surface variants: {same_entity_surface_variants}")
    print(f"[REVIEW] Report: {args.output / 'human_eval.json'}")
    print(f"[REVIEW] Pair labels: {args.output / 'human_gold_pairs.jsonl'}")


def serve(args) -> None:
    output_dir = args.output.resolve()
    handler = make_handler(output_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"[REVIEW] Open http://127.0.0.1:{args.port}/", flush=True)
    print(f"[REVIEW] Annotations autosave to {output_dir / 'annotations.json'}", flush=True)
    print("[REVIEW] Bound to loopback only; this server is not exposed to your network.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[REVIEW] Server stopped. Saved annotations are preserved.", flush=True)
    finally:
        server.server_close()


def build_parser():
    parser = argparse.ArgumentParser(description="Blind manual entity-linking review on anonymized source text")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Select a challenge/control sample and generate mention suggestions")
    prep.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    prep.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    prep.add_argument("--challenges", type=Path, default=DEFAULT_CHALLENGES)
    prep.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prep.add_argument("--hard-docs", type=int, default=6)
    prep.add_argument("--rejected-docs", type=int, default=5)
    prep.add_argument("--control-docs", type=int, default=3)
    prep.add_argument("--seed", type=int, default=4242)
    prep.add_argument("--max-chars", type=int, default=30000)
    prep.add_argument("--max-rejected-chars", type=int, default=6000,
                      help="Maximum original-text length for automation-rejected challenge docs")
    prep.add_argument("--replace-empty", action="store_true",
                      help="Replace an existing queue only if annotations.json contains no labels")
    prep.set_defaults(func=prepare)

    build = commands.add_parser("build-queue", help="Build the blind local queue from Kaggle NER offsets")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--predictions-file", type=Path)
    build.set_defaults(func=build_queue)

    web = commands.add_parser("serve", help="Run the local annotation UI")
    web.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    web.add_argument("--port", type=int, default=8765)
    web.set_defaults(func=serve)

    scoring = commands.add_parser("evaluate", help="Score transparent heuristics against completed human clusters")
    scoring.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    scoring.set_defaults(func=evaluate)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        if min(args.hard_docs, args.rejected_docs, args.control_docs) < 0:
            parser.error("document counts cannot be negative")
        if args.max_chars < 1000 or args.max_rejected_chars < 1000:
            parser.error("max source lengths must be at least 1000 characters")
    elif args.command == "build-queue":
        if args.predictions_file is None:
            args.predictions_file = args.output / "ner_predictions.jsonl"
    elif args.port < 1 or args.port > 65535:
        parser.error("port must be between 1 and 65535")
    args.func(args)


if __name__ == "__main__":
    main()
