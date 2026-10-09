"""Tests for the original-anonymized-text V4 dataset builder."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vdt_anonymization.entity_linking.v4_dataset import (
    _clean_document,
    _collision_candidates,
    _make_collision_variant,
    build_dataset,
    build_natural_pairs,
)


def _span(text: str, surface: str, entity_id: str, label: str = "PER", start_at: int = 0) -> dict:
    start = text.index(surface, start_at)
    return {
        "entity_id": entity_id,
        "label": label,
        "start": start,
        "end": start + len(surface),
        "original": surface,
        "replacement": f"GENERATED {entity_id}",
    }


def _fixture(doc_id: str = "doc-1") -> dict:
    text = "Nguyen Van A met A. Tran Van B met B."
    replacements = [
        _span(text, "Nguyen Van A", "PER_1"),
        _span(text, "A", "PER_1", start_at=text.index("Nguyen Van A") + len("Nguyen Van A")),
        _span(text, "Tran Van B", "PER_2"),
        _span(text, "B", "PER_2", start_at=text.index("Tran Van B") + len("Tran Van B")),
    ]
    mapping = {
        "doc_id": doc_id,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entities": [
            {"entity_id": "PER_1", "label": "PER", "role": "plaintiff", "synthetic_value": "Secret One"},
            {"entity_id": "PER_2", "label": "PER", "role": "defendant", "synthetic_value": "Secret Two"},
        ],
        "replacements": replacements,
    }
    row = {
        "doc_name": doc_id,
        "category": "family",
        "instance_level": "first",
        "curation": {"primary_challenge": "short_alias"},
        "original_anonymized_markdown": text,
        "synthetic_markdown": "Secret One met Secret One. Secret Two met Secret Two.",
        "synthetic_reconstruction": {"private": "generated names"},
    }
    return {"doc_id": doc_id, "row": row, "map": mapping, "text": text}


class V4DatasetTests(unittest.TestCase):
    def test_collision_is_created_in_original_text_and_keeps_weak_ids(self):
        item = _fixture()
        candidate = _collision_candidates(item)[0]
        row, mapping, pairs = _make_collision_variant(
            item, candidate, 1, "test", context_chars=40, seed="unit-test"
        )
        self.assertEqual(row["input_variant"], "counterfactual_marker_collision_01")
        self.assertNotIn("synthetic_markdown", row)
        self.assertNotIn("synthetic_value", json.dumps(mapping))
        self.assertEqual([pair["target_linked"] for pair in pairs].count(1), 2)
        self.assertEqual([pair["target_linked"] for pair in pairs].count(0), 1)
        negative = next(pair for pair in pairs if pair["target_linked"] == 0)
        self.assertEqual(negative["mention_a"]["surface"], negative["mention_b"]["surface"])
        self.assertEqual(negative["difficulty"], "collision_same_surface_negative")
        self.assertNotIn('"entity_id":', json.dumps(pairs))
        self.assertEqual(mapping["source_sha256"], hashlib.sha256(
            row["original_anonymized_markdown"].encode("utf-8")
        ).hexdigest())

    def test_natural_pairs_use_anonymized_text_and_do_not_leak_role_or_generated_names(self):
        item = _fixture()
        pairs, _, exclusions = build_natural_pairs(
            item["doc_id"], "train", item["text"], item["map"], {},
            context_chars=40, max_pairs_per_document=20, seed="unit-test",
        )
        serialized = json.dumps(pairs, ensure_ascii=False)
        self.assertIn("Nguyen Van A", serialized)
        self.assertNotIn("Secret One", serialized)
        self.assertNotIn("plaintiff", serialized)
        self.assertNotIn('"entity_id":', serialized)
        self.assertTrue(all(pair["features"]["same_role"] == 0.0 for pair in pairs))
        self.assertEqual(exclusions, {})

    def test_v4_pairs_include_parent_path_features_and_location_hard_diagnostic(self):
        text = (
            "huyện Châu Thành, tỉnh Bến Tre. " + ("x" * 100)
            + " huyện Châu Thành, tỉnh Tiền Giang."
        )
        spans = []
        for entity_id, start_at in (("LOC_1", 0), ("LOC_2", 40)):
            start = text.index("Châu Thành", start_at)
            spans.append({
                "entity_id": entity_id,
                "label": "LOC",
                "start": start,
                "end": start + len("Châu Thành"),
                "original": "Châu Thành",
            })
        mapping = {
            "entities": [
                {"entity_id": "LOC_1", "label": "LOC"},
                {"entity_id": "LOC_2", "label": "LOC"},
            ],
            "replacements": spans,
        }

        pairs, hard, _ = build_natural_pairs(
            "locations", "test", text, mapping, {}, context_chars=40,
            max_pairs_per_document=20, seed="unit-test",
        )
        negative = next(pair for pair in pairs if pair["target_linked"] == 0)
        self.assertEqual(negative["features"]["conflicting_admin_path"], 1.0)
        self.assertTrue(any(pair["difficulty"] == "location_hierarchy_conflict" for pair in hard))

    def test_natural_pairs_exclude_weak_labels_that_contradict_location_rules(self):
        text = (
            "huyện Châu Thành, tỉnh Bến Tre. " + ("x" * 100)
            + " huyện Châu Thành, tỉnh Tiền Giang."
        )
        starts = [text.index("Châu Thành"), text.rindex("Châu Thành")]
        shared_mapping = {
            "entities": [{"entity_id": "LOC_1", "label": "LOC"}],
            "replacements": [
                {"entity_id": "LOC_1", "label": "LOC", "start": start,
                 "end": start + len("Châu Thành"), "original": "Châu Thành"}
                for start in starts
            ],
        }
        pairs, _, exclusions = build_natural_pairs(
            "conflicting-positive", "train", text, shared_mapping, {},
            context_chars=40, max_pairs_per_document=20, seed="unit-test",
        )
        self.assertFalse(any(pair["target_linked"] for pair in pairs))
        self.assertEqual(exclusions["weak_positive_blocked_by_location_rule"], 1)

        split_mapping = {
            "entities": [
                {"entity_id": "LOC_1", "label": "LOC"},
                {"entity_id": "LOC_2", "label": "LOC"},
            ],
            "replacements": [
                {"entity_id": f"LOC_{index}", "label": "LOC", "start": start,
                 "end": start + len("Châu Thành"), "original": "Châu Thành"}
                for index, start in enumerate(starts, 1)
            ],
        }
        pairs, _, exclusions = build_natural_pairs(
            "linked-negative", "train", text.replace("Tiền Giang", "Bến Tre"),
            split_mapping, {}, context_chars=40, max_pairs_per_document=20, seed="unit-test",
        )
        self.assertFalse(pairs)
        self.assertEqual(exclusions["weak_negative_linked_by_location_rule"], 1)

    def test_document_allowlist_removes_synthetic_fields(self):
        row = _fixture()["row"]
        cleaned = _clean_document(row, row["original_anonymized_markdown"])
        serialized = json.dumps(cleaned)
        self.assertNotIn("synthetic_markdown", serialized)
        self.assertNotIn("Secret One", serialized)
        self.assertEqual(cleaned["entity_linking_input"], row["original_anonymized_markdown"])

    def test_full_build_preserves_splits_and_writes_separate_challenge_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v3 = root / "v3"
            for folder in ("documents", "maps"):
                (v3 / folder).mkdir(parents=True)
            for split in ("train", "validation", "test"):
                item = _fixture(split)
                (v3 / "documents" / f"{split}.jsonl").write_text(
                    json.dumps(item["row"], ensure_ascii=False) + "\n", encoding="utf-8"
                )
                (v3 / "maps" / f"{split}.jsonl").write_text(
                    json.dumps(item["map"], ensure_ascii=False) + "\n", encoding="utf-8"
                )
            output = root / "v4"
            manifest = build_dataset(
                v3, output, context_chars=40, max_pairs_per_document=20,
                train_collision_rate=1.0, max_train_variants_per_document=1, seed="unit-test",
            )
            self.assertEqual(manifest["dataset_version"], "entity-linking-original-anonymized-v4-location-hierarchy")
            self.assertEqual(manifest["counts"]["train"]["documents"], 1)
            with (output / "pairs" / "test_challenge.jsonl").open(encoding="utf-8") as handle:
                test_challenge = list(handle)
            self.assertEqual(len(test_challenge), 3)
            test_doc = json.loads((output / "documents" / "test.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(test_doc["input_variant"], "original_anonymized")
            self.assertNotIn("synthetic_markdown", test_doc)
            self.assertTrue((output / "maps" / "test_challenge.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
