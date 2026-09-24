import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import Namespace
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vdt_anonymization.review.manual_linking import (
    analyze_map,
    build_queue,
    validate_annotation,
    write_json_atomic,
    write_jsonl_atomic,
)


class ManualLinkingTests(unittest.TestCase):
    def _span(self, text, needle, offset=0):
        start = text.index(needle, offset)
        return {"start": start, "end": start + len(needle), "original": needle}

    def test_analyze_map_finds_surface_and_entity_variation_without_exporting_ids(self):
        source = "Anh Nguyễn Văn A gặp chị Nguyễn Văn A."
        first = self._span(source, "Nguyễn Văn A")
        second = self._span(source, "Nguyễn Văn A", first["end"])
        mapping = {
            "entities": [
                {"entity_id": "P1", "label": "PER"},
                {"entity_id": "P2", "label": "PER"},
            ],
            "replacements": [
                {**first, "entity_id": "P1"},
                {**second, "entity_id": "P2"},
            ],
        }

        strata, candidates = analyze_map(mapping, source)

        self.assertIn("same_surface_multiple_entities", strata)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["text"], "Nguyễn Văn A")
        self.assertNotIn("P1", repr(candidates))

    def test_validate_annotation_requires_resolved_decisions_for_complete_doc(self):
        doc = {
            "doc_id": "D1",
            "text": "An met An.",
            "mentions": [
                {"id": "M0001", "start": 0, "end": 2, "text": "An"},
                {"id": "M0002", "start": 7, "end": 9, "text": "An"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "still need a decision"):
            validate_annotation({"doc_id": "D1", "status": "complete"}, doc)

    def test_validate_annotation_checks_manual_span_against_original_text(self):
        doc = {"doc_id": "D1", "text": "An met Bình.", "mentions": []}
        with self.assertRaisesRegex(ValueError, "does not match source"):
            validate_annotation({
                "doc_id": "D1",
                "extra_mentions": [{"id": "X0001", "start": 7, "end": 11, "text": "Binh"}],
            }, doc)

    def test_validate_annotation_accepts_linked_group_and_singleton(self):
        doc = {
            "doc_id": "D1",
            "text": "An met An.",
            "mentions": [
                {"id": "M0001", "start": 0, "end": 2, "text": "An"},
                {"id": "M0002", "start": 7, "end": 9, "text": "An"},
            ],
        }
        result = validate_annotation({
            "doc_id": "D1",
            "status": "complete",
            "decisions": {
                "M0001": {"decision": "linked", "cluster_id": "E01"},
                "M0002": {"decision": "singleton"},
            },
        }, doc)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["decisions"]["M0001"]["cluster_id"], "E01")

    def test_build_queue_strips_ner_labels_scores_and_weak_map_metadata(self):
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            text = "Anh Nguyễn Văn A and chị A."
            write_jsonl_atomic(output / "kaggle_ner_input.jsonl", [
                {"doc_id": "D1", "text": text},
            ])
            write_json_atomic(output / "selection_manifest.json", {
                "counts": {"documents": 1},
                "private_selection": [{
                    "doc_id": "D1",
                    "sampling_group": "heldout_hard",
                    "selection_strata": ["hidden_stratum"],
                    "map_spans": [{
                        "start": text.index("Nguyễn Văn A"),
                        "end": text.index("Nguyễn Văn A") + len("Nguyễn Văn A"),
                        "text": "Nguyễn Văn A",
                    }],
                }],
            })
            predictions = output / "predictions.jsonl"
            write_jsonl_atomic(predictions, [{
                "doc_id": "D1",
                "entities": [{
                    "start": text.rindex("A"), "end": text.rindex("A") + 1,
                    "text": "A", "label": "PER", "score": 0.999,
                }],
            }])

            build_queue(Namespace(output=output, predictions_file=predictions))
            row = json.loads((output / "queue.jsonl").read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(set(row), {"doc_id", "text", "mentions"})
            self.assertEqual(len(row["mentions"]), 2)
            self.assertNotIn("label", repr(row))
            self.assertNotIn("score", repr(row))
            self.assertNotIn("hidden_stratum", repr(row))

if __name__ == "__main__":
    unittest.main()
