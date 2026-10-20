"""Evaluate the saved pair linker with a post-score name rule and context ablations.

This uses the saved 2026-10-16 checkpoint, weak reconstruction-map labels, and
document-disjoint validation/test challenge splits. It does not train a model.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from itertools import combinations, zip_longest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


if Path("/kaggle/input").is_dir():
    _checkpoints = list(Path("/kaggle/input").rglob("best_model.pt"))
    if len(_checkpoints) != 1:
        raise RuntimeError(f"Expected one mounted best_model.pt, found {len(_checkpoints)}")
    _model_dir = _checkpoints[0].parent
    _artifact_root = _model_dir.parent
    _source_zip = _artifact_root / "runtime_source.zip"
    if not _source_zip.is_file():
        _source_candidates = list(Path("/kaggle/input").rglob("source.zip"))
        if len(_source_candidates) == 1:
            _source_zip = _source_candidates[0]
    _source_dir = Path("/kaggle/working/vdt_runtime_source")
    _source_dir.mkdir(parents=True, exist_ok=True)
    if _source_zip.is_file():
        print(f"Source archive found: {_source_zip}", flush=True)
        with zipfile.ZipFile(_source_zip) as _archive:
            _archive.extractall(_source_dir)
    else:
        print(f"No source archive; input roots: {[p.name for p in Path('/kaggle/input').iterdir()]}", flush=True)
    _package_dirs = [p for p in Path("/kaggle/input").rglob("vdt_anonymization") if p.is_dir()]
    if _package_dirs:
        _source_dir = _package_dirs[0].parent
    print(f"Source import path: {_source_dir}; entries: {[p.name for p in _source_dir.iterdir()][:8]}", flush=True)
    ROOT = Path("/kaggle/working")
    DEFAULT_CHECKPOINT = _checkpoints[0]
    DEFAULT_TOKENIZER = _model_dir / "tokenizer"
    DEFAULT_DATA = _artifact_root / "challenge_dataset"
    DEFAULT_OUTPUT = ROOT / "previous_model_rule_context_experiment"
    sys.path.insert(0, str(_source_dir))
else:
    ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_DATA = ROOT / "kaggle_run" / "hybrid_rule_first_data_20261020"
    DEFAULT_OUTPUT = ROOT / "outputs" / "previous_model_rule_context_experiment_20261020"
    DEFAULT_CHECKPOINT = Path.home() / "AppData/Local/Temp/vdt_linker_rule_test_20261020/best_model.pt"
    DEFAULT_TOKENIZER = (
        Path.home()
        / "AppData/Local/Temp/vdt_linker_rule_test_20261020/entity_linker_independent_hardcases_20261016/pairwise_model/tokenizer"
    )
    sys.path.insert(0, str(ROOT / "src"))
from vdt_anonymization.entity_linking.clustering import (  # noqa: E402
    aggregate_cluster_metrics,
    cluster_metrics,
    complete_link_clusters,
)
from vdt_anonymization.entity_linking.dataset import normalize_text  # noqa: E402
from vdt_anonymization.entity_linking.location_constraints import location_rule_decision  # noqa: E402
from vdt_anonymization.entity_linking.raw_dataset import (  # noqa: E402
    RAW_PAIR_FEATURE_NAMES,
    _enrich,
    _target_replacements,
    raw_pair_features,
)
from vdt_anonymization.entity_linking.training import EntityLinkingModel  # noqa: E402


PAIR_THRESHOLD = 0.8
DOCUMENT_THRESHOLD = 0.1  # selected on natural validation documents in the saved run
CONTEXT_VARIANTS = ("window160", "window320", "paragraph")
RULE_CUTOFFS = (0.0, 0.15, 0.20, 0.25, 0.30, 0.40)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if label and predicted:
            tp += 1
        elif label:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + tn + fn
    return {
        "threshold": threshold,
        "examples": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def grouped_pair_metrics(rows: list[dict], scores: list[float], threshold: float) -> dict:
    result = {"overall": binary_metrics([int(r["target_linked"]) for r in rows], scores, threshold)}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row.get("difficulty") or "Unknown")].append(i)
    result["by_difficulty"] = {
        name: binary_metrics([int(rows[i]["target_linked"]) for i in indices],
                             [scores[i] for i in indices], threshold)
        for name, indices in sorted(groups.items())
    }
    return result


def make_context(text: str, mention: dict, variant: str) -> str:
    start, end = int(mention["start"]), int(mention["end"])
    surface = text[start:end]
    if variant in {"window160", "window320"}:
        radius = 160 if variant == "window160" else 320
        left, right = max(0, start - radius), min(len(text), end + radius)
    elif variant == "paragraph":
        left_break = text.rfind("\n\n", 0, start)
        right_break = text.find("\n\n", end)
        left = left_break + 2 if left_break >= 0 else 0
        right = right_break if right_break >= 0 else len(text)
        # Keep longer sections centered on the mention so tokenizer truncation
        # cannot drop the marked name or all its nearby context.
        cap = 900
        if end - left > cap // 2:
            left = max(left, start - cap // 2)
        if right - start > cap // 2:
            right = min(right, end + cap // 2)
        if right - left > cap:
            left, right = max(0, start - cap // 2), min(len(text), end + cap // 2)
    else:
        raise ValueError(f"unknown context variant: {variant}")
    return text[left:start] + "[MENTION]" + surface + "[/MENTION]" + text[end:right]


class Scorer:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path, threads: int):
        if not torch.cuda.is_available():
            torch.set_num_threads(max(1, threads))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.threshold = float(checkpoint["threshold"])
        self.config = checkpoint["config"]
        self.feature_names = tuple(self.config.get("feature_names") or RAW_PAIR_FEATURE_NAMES)
        self.max_length = int(self.config.get("max_length", 256))
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        encoder_config = AutoConfig.from_pretrained(self.config["model_name"])
        encoder = AutoModel.from_config(encoder_config)
        encoder.resize_token_embeddings(len(self.tokenizer))
        self.model = EntityLinkingModel(
            encoder,
            len(self.feature_names),
            float((self.config.get("arguments") or {}).get("dropout", 0.15)),
        )
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        del checkpoint
        gc.collect()

    @torch.inference_mode()
    def encode(self, contexts: list[str], batch_size: int = 24) -> torch.Tensor:
        chunks = []
        for begin in range(0, len(contexts), batch_size):
            encoded = self.tokenizer(
                contexts[begin:begin + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                chunks.append(self.model.encode(encoded))
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 768))

    @torch.inference_mode()
    def score_pairs(self, rows: list[dict], texts: dict[str, str], variant: str,
                    batch_size: int = 2048) -> list[float]:
        mention_by_key = {}
        pair_keys = []
        for row in rows:
            doc_id = str(row["doc_id"])
            text = texts[doc_id]
            keys = []
            for side in ("mention_a", "mention_b"):
                mention = row[side]
                key = (doc_id, int(mention["start"]), int(mention["end"]))
                if text[key[1]:key[2]] != str(mention["surface"]):
                    raise ValueError(f"span mismatch {doc_id} {key[1]}:{key[2]}")
                if key not in mention_by_key:
                    mention_by_key[key] = {
                        **mention,
                        "context": row[side]["context"] if variant == "window160"
                        else make_context(text, mention, variant),
                    }
                keys.append(key)
            pair_keys.append(tuple(keys))
        ordered_keys = list(mention_by_key)
        embeddings = self.encode([mention_by_key[k]["context"] for k in ordered_keys])
        key_to_index = {key: i for i, key in enumerate(ordered_keys)}
        pair_indices = [(key_to_index[a], key_to_index[b]) for a, b in pair_keys]
        scores = []
        for begin in range(0, len(rows), batch_size):
            end = min(len(rows), begin + batch_size)
            batch_pairs = pair_indices[begin:end]
            features = torch.tensor(
                [[float(row.get("features", {}).get(name, 0.0)) for name in self.feature_names]
                 for row in rows[begin:end]], dtype=torch.float32,
            )
            left = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long, device=self.device)
            right = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model.classify_embeddings(
                    embeddings[left], embeddings[right], features.to(self.device)
                )
            scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        return [float(x) for x in scores]

    @torch.inference_mode()
    def score_document(self, text: str, mentions: list[dict], batch_size: int = 2048) -> dict:
        mentions = sorted(mentions, key=lambda m: (int(m["start"]), int(m["end"]), str(m["label"])))
        contexts = [make_context(text, m, "window160") for m in mentions]
        embeddings = self.encode(contexts)
        pairs = [
            (left, right)
            for left, right in combinations(range(len(mentions)), 2)
            if str(mentions[left].get("label", "")).casefold()
            == str(mentions[right].get("label", "")).casefold()
        ]
        scores = {}
        cannot_link = set()
        must_link = set()
        for begin in range(0, len(pairs), batch_size):
            current = pairs[begin:begin + batch_size]
            features = torch.tensor(
                [[float(raw_pair_features(text, mentions[i], mentions[j]).get(name, 0.0))
                  for name in self.feature_names] for i, j in current],
                dtype=torch.float32,
            )
            left = torch.tensor([p[0] for p in current], dtype=torch.long, device=self.device)
            right = torch.tensor([p[1] for p in current], dtype=torch.long, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model.classify_embeddings(
                    embeddings[left], embeddings[right], features.to(self.device)
                )
            probabilities = torch.sigmoid(logits).float().cpu().tolist()
            for (i, j), probability in zip(current, probabilities):
                scores[(i, j)] = float(probability)
                decision = location_rule_decision({"mention_a": mentions[i], "mention_b": mentions[j]})
                if decision == "block":
                    cannot_link.add((i, j))
                elif decision == "link":
                    must_link.add((i, j))
        return {"mentions": mentions, "scores": scores, "cannot": cannot_link,
                "must": must_link, "gold": [str(m["gold_entity_id"]) for m in mentions]}


def name_tokens(surface: str) -> set[str]:
    return {t for t in re.findall(r"\w+", normalize_text(surface), flags=re.UNICODE) if t}


def full_name_conflicts(mentions: list[dict], cutoff: float) -> tuple[set[tuple[int, int]], list[dict]]:
    blocked, details = set(), []
    for i, j in combinations(range(len(mentions)), 2):
        a, b = mentions[i], mentions[j]
        if str(a.get("label", "")).upper() != "PER" or str(b.get("label", "")).upper() != "PER":
            continue
        ta, tb = name_tokens(str(a.get("surface") or "")), name_tokens(str(b.get("surface") or ""))
        if len(ta) < 2 or len(tb) < 2 or ta == tb or any(len(t) == 1 for t in ta | tb):
            continue
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard <= cutoff:
            blocked.add((i, j))
            details.append({"i": i, "j": j, "jaccard": jaccard,
                            "surface_a": a.get("surface"), "surface_b": b.get("surface")})
    return blocked, details


def score_documents(scorer: Scorer, docs_path: Path, maps_path: Path) -> list[dict]:
    scored = []
    with docs_path.open(encoding="utf-8-sig") as doc_h, maps_path.open(encoding="utf-8-sig") as map_h:
        for line_number, (doc_line, map_line) in enumerate(zip_longest(doc_h, map_h), 1):
            if doc_line is None or map_line is None:
                raise ValueError(f"document/map count mismatch at line {line_number}")
            row, mapping = json.loads(doc_line), json.loads(map_line)
            doc_id = str(row.get("doc_name") or row.get("case_id") or row.get("id") or "")
            if doc_id != str(mapping.get("doc_id") or ""):
                raise ValueError(f"document/map ID mismatch at line {line_number}")
            source = str(row.get("original_anonymized_markdown") or "")
            text = str(row.get("entity_linking_input") or row.get("synthetic_markdown") or "")
            if not doc_id or not text:
                raise ValueError(f"missing document ID/text at line {line_number}")
            entities = {str(e["entity_id"]): e for e in mapping.get("entities", [])}
            translated = _target_replacements(source, text, mapping)
            mentions = []
            for item in translated:
                entity = entities[str(item["entity_id"])]
                mention = _enrich(text, item, entity, 160)
                mention["gold_entity_id"] = str(item["entity_id"])
                mentions.append(mention)
            item = scorer.score_document(text, mentions)
            item["doc_id"] = doc_id
            scored.append(item)
            if len(scored) % 25 == 0:
                print(f"  scored {len(scored)} documents", flush=True)
    return scored


def split_file(data_dir: Path, split: str, kind: str) -> Path:
    local_name = f"{split}_independent_challenge_{kind}.jsonl"
    kaggle_name = f"{split}_independent_challenge.jsonl"
    candidates = (data_dir / local_name, data_dir / kind / kaggle_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find {split} {kind} data under {data_dir}")


def evaluate_document_rules(scored_docs: list[dict], threshold: float, cutoff: float | None) -> dict:
    per_doc, exact = [], 0
    candidate_pairs = blocked_count = blocked_positive = blocked_negative = 0
    changed_examples = []
    for doc in scored_docs:
        blocked = set()
        details = []
        if cutoff is not None:
            blocked, details = full_name_conflicts(doc["mentions"], cutoff)
        combined_cannot = set(doc["cannot"]) | blocked
        base_clusters = complete_link_clusters(doc["mentions"], doc["scores"], threshold,
                                               doc["cannot"], doc["must"])
        clusters = complete_link_clusters(doc["mentions"], doc["scores"], threshold,
                                          combined_cannot, doc["must"])
        metrics = cluster_metrics(doc["gold"], clusters)
        per_doc.append(metrics)
        counts = metrics["pairwise_counts"]
        exact += int(counts["fp"] == 0 and counts["fn"] == 0)
        candidate_pairs += len(doc["scores"])
        blocked_count += len(blocked)
        blocked_positive += sum(doc["gold"][d["i"]] == doc["gold"][d["j"]] for d in details)
        blocked_negative += len(details) - sum(doc["gold"][d["i"]] == doc["gold"][d["j"]] for d in details)
        if cutoff is not None:
            base_cluster_of = {i: cid for cid, group in enumerate(base_clusters) for i in group}
            rule_cluster_of = {i: cid for cid, group in enumerate(clusters) for i in group}
            details.sort(key=lambda d: doc["scores"].get((d["i"], d["j"]), 0.0), reverse=True)
            for detail in details:
                i, j = detail["i"], detail["j"]
                score = doc["scores"].get((i, j), 0.0)
                if base_cluster_of.get(i) == base_cluster_of.get(j) and rule_cluster_of.get(i) != rule_cluster_of.get(j):
                    changed_examples.append({
                        "doc_id": doc["doc_id"],
                        "name_a": doc["mentions"][i]["surface"],
                        "name_b": doc["mentions"][j]["surface"],
                        "pair_score": round(score, 6),
                        "name_token_jaccard": round(detail["jaccard"], 4),
                        "same_weak_entity": doc["gold"][i] == doc["gold"][j],
                    })
                    if len(changed_examples) >= 30:
                        break
    result = aggregate_cluster_metrics(per_doc)
    result.update({
        "candidate_pairs_scored": candidate_pairs,
        "name_rule_blocked_pairs": blocked_count,
        "name_rule_blocked_weak_positives": blocked_positive,
        "name_rule_blocked_weak_negatives": blocked_negative,
        "exact_partition_documents": exact,
        "exact_partition_accuracy": exact / len(scored_docs) if scored_docs else 0.0,
    })
    if changed_examples:
        result["examples_changed_by_name_rule"] = changed_examples[:12]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("Loading the saved checkpoint on CPU...", flush=True)
    scorer = Scorer(args.checkpoint, args.tokenizer_dir, args.threads)
    result = {
        "model": "saved ELECTRA base pair linker from Kaggle 2026-10-16",
        "inference_device": str(scorer.device),
        "checkpoint_threshold": scorer.threshold,
        "pair_threshold": PAIR_THRESHOLD,
        "document_threshold": DOCUMENT_THRESHOLD,
        "supervision_warning": "Labels are weak reconstruction-map IDs, not verified human identities.",
        "pair_context_ablation": {},
        "document_rule_comparison": {},
    }

    text_by_split = {}
    pair_rows_by_split = {}
    for split in ("validation", "test"):
        docs = read_jsonl(split_file(args.data_dir, split, "documents"))
        texts = {str(d["doc_name"]): str(d.get("entity_linking_input") or d.get("synthetic_markdown") or "")
                 for d in docs}
        rows = read_jsonl(split_file(args.data_dir, split, "pairs"))
        text_by_split[split], pair_rows_by_split[split] = texts, rows
        result["pair_context_ablation"][split] = {}
        for variant in CONTEXT_VARIANTS:
            print(f"Scoring {split} pairs with {variant} context ({len(rows)} pairs)...", flush=True)
            scores = scorer.score_pairs(rows, texts, variant)
            metrics = grouped_pair_metrics(rows, scores, PAIR_THRESHOLD)
            metrics["context_definition"] = {
                "window160": "saved training context, 160 characters to either side",
                "window320": "320 characters to either side",
                "paragraph": "containing blank-line paragraph, centered and capped at 900 characters",
            }[variant]
            result["pair_context_ablation"][split][variant] = metrics
            if variant == "window160":
                result["pair_context_ablation"][split]["window160"]["scores_for_rule_check"] = scores
        # Store only metrics; keep model inference outputs small in the final JSON.
        result["pair_context_ablation"][split].pop("window160_scores", None)

    # Score every mention pair in the full validation/test documents once. The
    # validation split selects only the name-rule Jaccard cutoff; the model and
    # saved document threshold stay fixed.
    doc_scores = {}
    for split in ("validation", "test"):
        docs_path = split_file(args.data_dir, split, "documents")
        maps_path = split_file(args.data_dir, split, "maps")
        print(f"Scoring all mentions in {split} documents...", flush=True)
        doc_scores[split] = score_documents(scorer, docs_path, maps_path)

    validation_results = {}
    for cutoff in (None, *RULE_CUTOFFS):
        name = "model_only" if cutoff is None else f"full_name_jaccard_le_{cutoff:.2f}"
        validation_results[name] = evaluate_document_rules(doc_scores["validation"], DOCUMENT_THRESHOLD, cutoff)
    best_name = max(
        (name for name in validation_results if name != "model_only"),
        key=lambda name: (
            validation_results[name]["b_cubed"]["f1"],
            validation_results[name]["b_cubed"]["precision"],
            validation_results[name]["pairwise"]["precision"],
        ),
    )
    best_cutoff = float(best_name.rsplit("_", 1)[1])
    result["document_rule_comparison"] = {
        "rule": "During complete-link clustering, block links between two distinct PER full names when token-set Jaccard overlap is at or below the selected cutoff. One-token aliases and names containing initials are left to the model.",
        "selection": "Choose the cutoff by B-cubed F1 on the validation challenge documents at the previously selected 0.10 document threshold.",
        "validation_threshold": DOCUMENT_THRESHOLD,
        "validation_cutoff_sweep": validation_results,
        "selected_cutoff": best_cutoff,
        "test_model_only": evaluate_document_rules(doc_scores["test"], DOCUMENT_THRESHOLD, None),
        "test_model_plus_name_rule": evaluate_document_rules(doc_scores["test"], DOCUMENT_THRESHOLD, best_cutoff),
    }
    # Remove raw pair scores from result and report runtime.
    for split in ("validation", "test"):
        result["pair_context_ablation"][split].pop("window160_scores_for_rule_check", None)
        result["pair_context_ablation"][split].pop("scores_for_rule_check", None)
    result["runtime_seconds"] = round(time.time() - start, 1)
    result_path = args.output_dir / "experiment_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {result_path}", flush=True)
    print(json.dumps({
        "runtime_seconds": result["runtime_seconds"],
        "pair_test": {k: v["overall"] for k, v in result["pair_context_ablation"]["test"].items()},
        "document_validation_model_only": validation_results["model_only"]["b_cubed"],
        "selected_rule": best_name,
        "document_test_model_only": result["document_rule_comparison"]["test_model_only"]["b_cubed"],
        "document_test_model_plus_rule": result["document_rule_comparison"]["test_model_plus_name_rule"]["b_cubed"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
