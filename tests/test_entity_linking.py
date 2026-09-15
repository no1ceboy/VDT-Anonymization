"""Tests for entity-linking split, examples, baselines, and model utilities."""

import json
import unittest

import torch
from torch import nn

from vdt_anonymization.entity_linking.dataset import (
    MENTION_CLOSE,
    MENTION_OPEN,
    allocate_splits,
    build_document_pairs,
    pair_features,
)
from vdt_anonymization.entity_linking.baseline import predict
from vdt_anonymization.entity_linking.training import EntityLinkingModel, binary_metrics


def replacement(text, surface, occurrence, entity_id, label="PER"):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(surface, start + 1)
    return {"entity_id": entity_id, "label": label, "start": start, "end": start + len(surface), "original": surface}


class EntityLinkingDataTests(unittest.TestCase):
    def test_document_split_is_deterministic_and_disjoint(self):
        items = [
            {
                "doc_id": str(index),
                "row": {"category": "civil", "instance_level": "first", "curation": {"primary_challenge": "H1"}},
            }
            for index in range(100)
        ]
        first = allocate_splits(items, 0.1, 0.1, "seed")
        second = allocate_splits(list(reversed(items)), 0.1, 0.1, "seed")
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("train"), 80)
        self.assertEqual(list(first.values()).count("validation"), 10)
        self.assertEqual(list(first.values()).count("test"), 10)

    def test_rare_challenge_gets_held_out_coverage(self):
        items = [
            {"doc_id": str(index), "row": {"category": f"category-{index}", "instance_level": "first",
                                             "curation": {"primary_challenge": "rare"}}}
            for index in range(3)
        ]
        assignments = allocate_splits(items, 0.1, 0.1, "seed")
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_pairs_use_source_context_and_not_synthetic_values(self):
        text = "Nguyễn Văn A gặp A. Bùi Văn B gặp B."
        row = {"doc_name": "case-1", "original_anonymized_markdown": text}
        mapping = {
            "entities": [
                {"entity_id": "PER_1", "label": "PER", "marker": "A", "role": "person", "synthetic_value": "SECRET ONE"},
                {"entity_id": "PER_2", "label": "PER", "marker": "B", "role": "person", "synthetic_value": "SECRET TWO"},
            ],
            "replacements": [
                replacement(text, "Nguyễn Văn A", 0, "PER_1"),
                replacement(text, "A", 1, "PER_1"),
                replacement(text, "Bùi Văn B", 0, "PER_2"),
                replacement(text, "B", 2, "PER_2"),
            ],
        }
        pairs = build_document_pairs(row, mapping, "train", 24, 8, 1.0, 1, 20, "seed")
        self.assertTrue(any(pair["target_linked"] == 1 for pair in pairs))
        self.assertTrue(any(pair["target_linked"] == 0 for pair in pairs))
        serialized = json.dumps(pairs, ensure_ascii=False)
        self.assertNotIn("SECRET", serialized)
        self.assertIn(MENTION_OPEN, serialized)
        self.assertIn(MENTION_CLOSE, serialized)
        self.assertEqual(pairs[0]["document_metadata"]["category"], "Unknown")
        for pair in pairs:
            expected = pair["mention_a"]["entity_id"] == pair["mention_b"]["entity_id"]
            self.assertEqual(bool(pair["target_linked"]), expected)

    def test_pair_features_and_rule_baseline(self):
        base = {"label": "PER", "surface": "H1", "marker": "H1", "marker_family": "h", "role": "person", "start": 10}
        other = {"label": "PER", "surface": "H2", "marker": "H2", "marker_family": "h", "role": "person", "start": 20}
        features = pair_features(base, other)
        self.assertEqual(features["same_marker"], 0.0)
        self.assertEqual(features["same_marker_family"], 1.0)
        pair = {"mention_a": base, "mention_b": other}
        self.assertEqual(predict(pair, "exact_marker"), 0)
        self.assertEqual(predict(pair, "marker_family"), 1)

    def test_binary_metrics(self):
        result = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
        self.assertEqual(result["confusion"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(result["f1"], 0.5)

    def test_pair_model_shape_with_fake_encoder(self):
        class Encoder(nn.Module):
            class Config:
                hidden_size = 4
            config = Config()

            def forward(self, input_ids, attention_mask):
                class Output:
                    pass
                output = Output()
                output.last_hidden_state = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
                return output

        model = EntityLinkingModel(Encoder(), feature_count=2, dropout=0.0)
        batch = {"input_ids": torch.tensor([[1, 2, 0], [2, 3, 1]]), "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
        logits = model(batch, batch, torch.zeros((2, 2)))
        self.assertEqual(tuple(logits.shape), (2,))


if __name__ == "__main__":
    unittest.main()
