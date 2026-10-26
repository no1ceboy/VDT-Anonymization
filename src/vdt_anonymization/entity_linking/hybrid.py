"""Rule-constrained, lightweight pair scorer for document entity linking.

The model is intentionally small: hand-built name/context signals feed a
logistic pair scorer, then the existing location rules and complete-link
clustering enforce document-level constraints.  It does not use transformer
embeddings or any challenge/test identity metadata.

Training needs scikit-learn.  The exported JSON weights can be loaded at
inference time without scikit-learn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations, zip_longest
from pathlib import Path
from typing import Iterable

from .clustering import aggregate_cluster_metrics, cluster_metrics, complete_link_clusters
from .dataset import normalize_text
from .location_constraints import location_rule_decision
from .raw_dataset import _enrich, _target_replacements, raw_pair_features


MARK_OPEN = "[MENTION]"
MARK_CLOSE = "[/MENTION]"
TOKEN_RE = re.compile(r"[\wÀ-ỹĐđ]+", flags=re.UNICODE)
TITLE_WORDS = {"anh", "chi", "ong", "ba", "co", "chu", "bac", "em", "thay", "cu", "di", "mo", "cau"}
TITLE_GENDER = {
    "anh": "male", "ong": "male", "chu": "male", "cau": "male", "thay": "male",
    "chi": "female", "ba": "female", "co": "female", "di": "female", "mo": "female",
}
ROLE_PATTERNS = {
    "plaintiff": re.compile(r"nguyên\s+đơn|người\s+khởi\s+kiện", re.I),
    "defendant": re.compile(r"bị\s+đơn|người\s+bị\s+kiện", re.I),
    "accused": re.compile(r"bị\s+cáo", re.I),
    "victim": re.compile(r"bị\s+hại|người\s+bị\s+hại", re.I),
    "witness": re.compile(r"người\s+làm\s+chứng|người\s+chứng\s+kiến", re.I),
    "related_person": re.compile(r"người\s+có\s+quyền\s+lợi.{0,24}liên\s+quan", re.I),
}
STOP_WORDS = {
    "va", "voi", "cua", "cho", "tai", "theo", "la", "co", "anh", "chi", "ong", "ba",
    "em", "bi", "nguyen", "don", "cao", "so", "ngay", "thang", "nam", "da", "duoc",
    "trong", "tren", "duoi", "mot", "cac", "nhung", "den", "tu", "ve", "khi", "thi",
}


def _fold(value: object) -> str:
    text = normalize_text(value)
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").replace("đ", "d")


def _tokens(value: object) -> list[str]:
    return [token for token in TOKEN_RE.findall(_fold(value)) if token]


def _marked_context(mention: dict) -> tuple[str, str, str]:
    context = str(mention.get("context") or "")
    if MARK_OPEN in context and MARK_CLOSE in context:
        left, remainder = context.split(MARK_OPEN, 1)
        marked, right = remainder.split(MARK_CLOSE, 1)
        return left, marked, right
    surface = str(mention.get("surface") or "")
    index = context.find(surface) if surface else -1
    if index < 0:
        return context, surface, ""
    return context[:index], surface, context[index + len(surface):]


def _nearest_title(left: str) -> str:
    tokens = _tokens(left[-32:])
    return tokens[-1] if tokens and tokens[-1] in TITLE_WORDS else "unknown"


def _conflicting_gendered_titles(left: dict, right: dict) -> bool:
    """Return true only when both mentions have clear, opposing title cues."""
    title_left = _nearest_title(_marked_context(left)[0])
    title_right = _nearest_title(_marked_context(right)[0])
    gender_left = TITLE_GENDER.get(title_left)
    gender_right = TITLE_GENDER.get(title_right)
    return gender_left is not None and gender_right is not None and gender_left != gender_right


def _nearest_role(left: str, right: str) -> str:
    window = left[-130:] + " " + right[:60]
    matches = []
    for role, pattern in ROLE_PATTERNS.items():
        for match in pattern.finditer(window):
            matches.append((abs(len(left[-130:]) - match.end()), role))
    return min(matches)[1] if matches else "unknown"


def _age_signal(left: str, right: str) -> str:
    window = left[-70:] + " " + right[:70]
    match = re.search(r"(?:sinh\s+năm|năm\s+sinh|sinh\s+ngày)[^\d]{0,8}(19\d{2}|20\d{2})", window, re.I)
    return match.group(1) if match else "unknown"


def pair_features(pair: dict) -> dict[str, float]:
    """Create interpretable lexical, mention-context, and legal-role features."""
    a, b = pair["mention_a"], pair["mention_b"]
    features: dict[str, float] = {
        f"base:{key}": float(value)
        for key, value in (pair.get("features") or {}).items()
        if isinstance(value, (int, float))
    }
    surface_a, surface_b = str(a.get("surface") or ""), str(b.get("surface") or "")
    norm_a, norm_b = _fold(surface_a), _fold(surface_b)
    tokens_a, tokens_b = _tokens(surface_a), _tokens(surface_b)
    set_a, set_b = set(tokens_a), set(tokens_b)
    common = set_a & set_b
    union = set_a | set_b
    features.update({
        "pair:same_type": float(str(a.get("label") or "").casefold() == str(b.get("label") or "").casefold()),
        "pair:normalized_surface_equal": float(bool(norm_a) and norm_a == norm_b),
        "pair:contains_short_form": float(bool(norm_a and norm_b) and (norm_a in norm_b or norm_b in norm_a)),
        "pair:token_jaccard": len(common) / max(1, len(union)),
        "pair:shared_token_count": float(len(common)),
        "pair:token_count_min": float(min(len(tokens_a), len(tokens_b))),
        "pair:token_count_max": float(max(len(tokens_a), len(tokens_b))),
        "pair:first_token_equal": float(bool(tokens_a and tokens_b and tokens_a[0] == tokens_b[0])),
        "pair:last_token_equal": float(bool(tokens_a and tokens_b and tokens_a[-1] == tokens_b[-1])),
        "pair:source_role_equal": float(bool(a.get("role")) and a.get("role") == b.get("role")),
        "pair:conflicting_gendered_titles": float(_conflicting_gendered_titles(a, b)),
    })
    for token in common:
        features[f"name:shared:{token}"] = 1.0
    if common:
        features[f"name:one_shared_token:{sorted(common)[0]}"] = 1.0

    left_a, _, right_a = _marked_context(a)
    left_b, _, right_b = _marked_context(b)
    local_a = _tokens(left_a[-80:] + " " + right_a[:80])
    local_b = _tokens(left_b[-80:] + " " + right_b[:80])
    local_set_a, local_set_b = set(local_a), set(local_b)
    local_common = local_set_a & local_set_b
    local_union = local_set_a | local_set_b
    features["context:local_jaccard"] = len(local_common) / max(1, len(local_union))
    features["context:local_overlap_count"] = float(len(local_common))
    title_a, title_b = _nearest_title(left_a), _nearest_title(left_b)
    role_a, role_b = _nearest_role(left_a, right_a), _nearest_role(left_b, right_b)
    age_a, age_b = _age_signal(left_a, right_a), _age_signal(left_b, right_b)
    features["context:title_equal"] = float(title_a != "unknown" and title_a == title_b)
    features["context:role_equal"] = float(role_a != "unknown" and role_a == role_b)
    features["context:age_equal"] = float(age_a != "unknown" and age_a == age_b)
    for side, title, role, age in (
        ("a", title_a, role_a, age_a),
        ("b", title_b, role_b, age_b),
    ):
        features[f"context:{side}:title:{title}"] = 1.0
        features[f"context:{side}:role:{role}"] = 1.0
        features[f"context:{side}:age:{age}"] = 1.0
    for token in local_common:
        if token not in STOP_WORDS and len(token) > 1:
            features[f"context:shared_word:{token}"] = 1.0
    # Directional local words let the scorer distinguish a name used in two
    # different party slots from one repeated in the same narrative context.
    for token in local_set_a - local_set_b:
        if token not in STOP_WORDS and len(token) > 1:
            features[f"context:a_only:{token}"] = 1.0
    for token in local_set_b - local_set_a:
        if token not in STOP_WORDS and len(token) > 1:
            features[f"context:b_only:{token}"] = 1.0
    return features


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _binary_metrics(labels: list[int], predictions: list[int]) -> dict:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pairs": len(labels), "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _threshold_for_f1(rows: list[dict], scores: list[float], model: "HybridPairScorer") -> float:
    labels = [int(row["target_linked"]) for row in rows]
    candidates = sorted({0.0, 1.0, *(round(score, 8) for score in scores)})
    return max(
        candidates,
        key=lambda threshold: (
            _binary_metrics(labels, [model.predict(row, threshold, score)
                                     for row, score in zip(rows, scores)])["f1"],
            _binary_metrics(labels, [model.predict(row, threshold, score)
                                     for row, score in zip(rows, scores)])["precision"],
            threshold,
        ),
    )


class HybridPairScorer:
    """JSON-backed small scorer; inference does not require scikit-learn."""

    def __init__(self, weights: dict[str, float], intercept: float,
                 pair_threshold: float, cluster_threshold: float,
                 validation: dict | None = None):
        self.weights = weights
        self.intercept = float(intercept)
        self.pair_threshold = float(pair_threshold)
        self.cluster_threshold = float(cluster_threshold)
        self.validation = validation or {}

    def score(self, pair: dict) -> float:
        logit = self.intercept + sum(
            float(value) * self.weights.get(name, 0.0)
            for name, value in pair_features(pair).items()
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-min(logit, 709.0)))
        exp_logit = math.exp(max(logit, -709.0))
        return exp_logit / (1.0 + exp_logit)

    def predict(self, pair: dict, threshold: float | None = None,
                score: float | None = None) -> int:
        a, b = pair["mention_a"], pair["mention_b"]
        if str(a.get("label") or "").casefold() != str(b.get("label") or "").casefold():
            return 0
        location = location_rule_decision(pair)
        if location == "block":
            return 0
        if location == "link":
            return 1
        if _conflicting_gendered_titles(a, b):
            return 0
        cutoff = self.pair_threshold if threshold is None else float(threshold)
        pair_score = self.score(pair) if score is None else float(score)
        return int(pair_score >= cutoff)

    def to_json(self) -> dict:
        return {
            "model": "vdt_rule_constrained_logistic_pair_scorer_v1",
            "input_features": "name token overlap, local marked context, nearby title/procedural role/age, existing pair features",
            "decision_policy": "entity-type gate; existing canonical location constraints; conflicting gendered-title block; logistic pair score; validation-selected pair/document thresholds; complete-link clustering",
            "intercept": self.intercept,
            "weights": self.weights,
            "pair_threshold": self.pair_threshold,
            "cluster_threshold": self.cluster_threshold,
            "validation": self.validation,
            "supervision_warning": "Fit and evaluated against weak reconstruction identity labels; not human gold.",
        }

    @classmethod
    def from_json(cls, path: Path) -> "HybridPairScorer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["weights"], data["intercept"], data["pair_threshold"],
                   data["cluster_threshold"], data.get("validation"))


def _score_pair_rows(model: HybridPairScorer, rows: list[dict]) -> list[float]:
    return [model.score(row) for row in rows]


def _document_rows(documents_path: Path, maps_path: Path, context_chars: int = 160) -> list[dict]:
    output = []
    with documents_path.open(encoding="utf-8-sig") as documents, maps_path.open(encoding="utf-8-sig") as maps:
        for line_number, (doc_line, map_line) in enumerate(zip_longest(documents, maps), 1):
            if doc_line is None or map_line is None:
                raise ValueError(f"document/map count mismatch on line {line_number}")
            if not doc_line.strip() or not map_line.strip():
                raise ValueError(f"blank document/map record on line {line_number}")
            row, mapping = json.loads(doc_line), json.loads(map_line)
            doc_id = str(row.get("doc_name") or row.get("case_id") or row.get("id") or "")
            if not doc_id or doc_id != str(mapping.get("doc_id") or ""):
                raise ValueError(f"document/map ID mismatch on line {line_number}")
            source = str(row.get("original_anonymized_markdown") or "")
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != mapping.get("source_sha256"):
                raise ValueError(f"source hash mismatch for document {doc_id}")
            text = str(row.get("entity_linking_input") or row.get("synthetic_markdown") or "")
            entities = {str(item["entity_id"]): item for item in mapping.get("entities", [])}
            translated = _target_replacements(source, text, mapping)
            mentions = []
            for item in translated:
                entity = entities.get(str(item["entity_id"]))
                if entity is None:
                    raise ValueError(f"unknown reconstruction entity in document {doc_id}")
                mention = _enrich(text, item, entity, context_chars)
                mention["gold_entity_id"] = str(item["entity_id"])
                mentions.append(mention)
            mentions.sort(key=lambda mention: (int(mention["start"]), int(mention["end"])))
            output.append({"doc_id": doc_id, "text": text, "mentions": mentions,
                           "gold_entity_ids": [item["gold_entity_id"] for item in mentions],
                           "challenge_document_type": row.get("entity_linking_challenge_document_type", "unknown")})
    return output


def _score_documents(model: HybridPairScorer, documents: list[dict]) -> list[dict]:
    scored = []
    for document in documents:
        mentions = document["mentions"]
        scores: dict[tuple[int, int], float] = {}
        must_link, cannot_link = set(), set()
        for left, right in combinations(range(len(mentions)), 2):
            a, b = mentions[left], mentions[right]
            if str(a.get("label") or "").casefold() != str(b.get("label") or "").casefold():
                continue
            pair = {"mention_a": a, "mention_b": b,
                    "features": raw_pair_features(document["text"], a, b)}
            location = location_rule_decision(pair)
            if location == "link":
                must_link.add((left, right))
            elif location == "block":
                cannot_link.add((left, right))
            elif _conflicting_gendered_titles(a, b):
                cannot_link.add((left, right))
            else:
                scores[(left, right)] = model.score(pair)
        scored.append({**document, "scores": scores, "must_link": must_link,
                       "cannot_link": cannot_link,
                       "candidate_pairs_scored": len(scores) + len(must_link) + len(cannot_link)})
    return scored


def _evaluate_documents(scored_documents: list[dict], threshold: float,
                        include_predictions: bool = False) -> tuple[dict, list[dict]]:
    metrics_by_document, exact_partitions = [], 0
    predicted_documents = []
    for item in scored_documents:
        clusters = complete_link_clusters(item["mentions"], item["scores"], threshold,
                                          item["cannot_link"], item["must_link"])
        measured = cluster_metrics(item["gold_entity_ids"], clusters)
        metrics_by_document.append(measured)
        counts = measured["pairwise_counts"]
        exact_partitions += int(counts["fp"] == 0 and counts["fn"] == 0)
        if include_predictions:
            owners = {}
            for cluster_id, members in enumerate(clusters):
                for index in members:
                    owners[index] = cluster_id
            predicted_documents.append({**item, "clusters": clusters, "owners": owners})
    result = aggregate_cluster_metrics(metrics_by_document)
    result["exact_partition_documents"] = exact_partitions
    result["exact_partition_accuracy"] = exact_partitions / len(scored_documents) if scored_documents else 0.0
    result["candidate_pairs_scored"] = sum(item["candidate_pairs_scored"] for item in scored_documents)
    return result, predicted_documents


def _pair_evaluation(rows: list[dict], scores: list[float], threshold: float,
                     model: HybridPairScorer | None = None) -> dict:
    labels = [int(item["target_linked"]) for item in rows]
    predictions = ([model.predict(row, threshold, score) for row, score in zip(rows, scores)] if model is not None
                   else [int(score >= threshold) for score in scores])
    overall = _binary_metrics(labels, predictions)
    grouped: dict[str, list[int]] = defaultdict(list)
    grouped_predictions: dict[str, list[int]] = defaultdict(list)
    for row, label, prediction in zip(rows, labels, predictions):
        key = str(row.get("difficulty") or "unknown")
        grouped[key].append(label)
        grouped_predictions[key].append(prediction)
    return {"threshold": threshold, "overall": overall,
            "by_challenge_category": {key: _binary_metrics(grouped[key], grouped_predictions[key])
                                       for key in sorted(grouped)}}


def train_and_evaluate(args: argparse.Namespace) -> dict:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:  # pragma: no cover - optional training dependency
        raise RuntimeError("Hybrid scorer training requires scikit-learn; install vdt-anonymization[hybrid].") from error

    train_rows, validation_rows, test_rows = map(_read_jsonl,
        (args.train_pairs, args.validation_pairs, args.test_pairs))
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("train, validation, and test pair files must all be nonempty")
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([pair_features(row) for row in train_rows])
    y_train = [int(row["target_linked"]) for row in train_rows]
    classifier = LogisticRegression(C=args.regularization, max_iter=1500, solver="liblinear", random_state=17)
    classifier.fit(x_train, y_train)
    validation_scores = classifier.predict_proba(
        vectorizer.transform([pair_features(row) for row in validation_rows]))[:, 1].tolist()
    validation_labels = [int(item["target_linked"]) for item in validation_rows]
    names = vectorizer.get_feature_names_out()
    model = HybridPairScorer(
        dict(zip(names, classifier.coef_[0].astype(float).tolist())),
        float(classifier.intercept_[0]), 0.5, 0.5,
    )
    pair_threshold = _threshold_for_f1(validation_rows, validation_scores, model)
    model.pair_threshold = pair_threshold

    validation_docs = _score_documents(model, _document_rows(args.validation_documents, args.validation_maps))
    cluster_threshold_candidates = [round(value, 2) for value in [x / 100 for x in range(5, 100, 5)]]
    validation_cluster_sweep = []
    for threshold in cluster_threshold_candidates:
        measured, _ = _evaluate_documents(validation_docs, threshold)
        validation_cluster_sweep.append({"threshold": threshold, "b_cubed": measured["b_cubed"],
                                         "pairwise": measured["pairwise"],
                                         "exact_partition_accuracy": measured["exact_partition_accuracy"]})
    selected_cluster = max(validation_cluster_sweep, key=lambda item: (
        item["b_cubed"]["f1"], item["b_cubed"]["precision"], item["pairwise"]["precision"], item["threshold"]))
    model.cluster_threshold = float(selected_cluster["threshold"])
    model.validation = {
        "pair_threshold_selected_by": "validation pair F1, precision tie break",
        "pair_threshold_metrics": _pair_evaluation(validation_rows, validation_scores, pair_threshold, model)["overall"],
        "cluster_threshold_selected_by": "validation document B-cubed F1, precision tie breaks",
        "cluster_threshold": model.cluster_threshold,
        "cluster_threshold_metrics": selected_cluster,
        "candidate_cluster_thresholds": validation_cluster_sweep,
    }

    test_scores = _score_pair_rows(model, test_rows)
    pair_report = _pair_evaluation(test_rows, test_scores, pair_threshold, model)
    test_documents = _document_rows(args.test_documents, args.test_maps)
    scored_test_docs = _score_documents(model, test_documents)
    document_metrics, document_predictions = _evaluate_documents(
        scored_test_docs, model.cluster_threshold, include_predictions=True)

    test_id_set = {str(item["doc_id"]) for item in test_documents}
    if any(str(row["doc_id"]) not in test_id_set for row in test_rows):
        raise ValueError("test pair file contains documents outside the document/map test split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hybrid_pair_scorer.json"
    model_path.write_text(json.dumps(model.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_predictions = []
    for row, score in zip(test_rows, test_scores):
        test_predictions.append({"pair_id": row.get("pair_id"), "doc_id": str(row.get("doc_id")),
                                 "category": row.get("difficulty"), "target_linked": int(row["target_linked"]),
                                 "score": score, "predicted_linked": model.predict(row, pair_threshold),
                                 "surface_a": row["mention_a"].get("surface"),
                                 "surface_b": row["mention_b"].get("surface")})
    report = {
        "evaluation_kind": "document-disjoint held-out reconstructed challenge; weak identity labels",
        "training": {"train_pairs": len(train_rows), "train_positive": sum(y_train),
                     "train_negative": len(y_train) - sum(y_train),
                     "validation_pairs": len(validation_rows), "test_pairs": len(test_rows),
                     "features": len(names), "regularization_C": args.regularization},
        "model": str(model_path),
        "pair_threshold": pair_threshold,
        "cluster_threshold": model.cluster_threshold,
        "validation": model.validation,
        "test": {"pairwise": pair_report, "document_clustering": document_metrics},
        "test_pair_predictions": test_predictions,
        "test_document_predictions": [
            {"doc_id": item["doc_id"], "text": item["text"],
             "mentions": [{"start": mention["start"], "end": mention["end"],
                           "surface": mention["surface"], "label": mention["label"],
                           "gold_entity_id": mention["gold_entity_id"],
                           "cluster": int(document["owners"][index])}
                          for index, mention in enumerate(item["mentions"])]}
            for item, document in zip(scored_test_docs, document_predictions)
        ],
        "warning": "All fit/validation/test labels in this comparison come from the reconstruction maps, not human adjudication.",
    }
    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--validation-pairs", required=True, type=Path)
    parser.add_argument("--test-pairs", required=True, type=Path)
    parser.add_argument("--validation-documents", required=True, type=Path)
    parser.add_argument("--validation-maps", required=True, type=Path)
    parser.add_argument("--test-documents", required=True, type=Path)
    parser.add_argument("--test-maps", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--regularization", type=float, default=0.5)
    args = parser.parse_args()
    for name in ("train_pairs", "validation_pairs", "test_pairs", "validation_documents",
                 "validation_maps", "test_documents", "test_maps"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    result = train_and_evaluate(args)
    print(json.dumps({"model": result["model"], "training": result["training"],
                      "pair_threshold": result["pair_threshold"],
                      "cluster_threshold": result["cluster_threshold"],
                      "pairwise": result["test"]["pairwise"]["overall"],
                      "document_clustering": result["test"]["document_clustering"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
