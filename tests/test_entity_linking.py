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
from vdt_anonymization.entity_linking.evaluate import _link_error_summary
from vdt_anonymization.entity_linking.training import EntityLinkingModel, binary_metrics
from vdt_anonymization.entity_linking.finetuning import normalize_finetune_mode, select_lora_targets
from vdt_anonymization.entity_linking.location_constraints import (
    explicit_province,
    has_explicit_province_conflict,
    location_province_features,
)
from vdt_anonymization.entity_linking.raw_dataset import raw_pair_features


def replacement(text, surface, occurrence, entity_id, label="PER"):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(surface, start + 1)
    return {"entity_id": entity_id, "label": label, "start": start, "end": start + len(surface), "original": surface}


class EntityLinkingDataTests(unittest.TestCase):
    def test_negative_only_collision_summary_reports_false_link_rate(self):
        labels = [0] * 23
        scores = [0.2] * 22 + [0.9]
        summary = _link_error_summary(labels, scores, 0.55)

        self.assertEqual(summary["different_entity_pairs"], 23)
        self.assertEqual(summary["false_links"], 1)
        self.assertEqual(summary["correctly_kept_separate"], 22)
        self.assertAlmostEqual(summary["false_link_rate"], 1 / 23)
        self.assertEqual(summary["same_entity_pairs"], 0)

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

    def test_conflicting_explicit_provinces_veto_location_link(self):
        a = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": "Cư trú tại: Ấp Minh Tân, xã Song Thuận, huyện [MENTION]Châu Thành[/MENTION], tỉnh Tiền Giang.",
        }
        b = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": "UBND xã An Phước, huyện [MENTION]Châu Thành[/MENTION], tỉnh Bến Tre;",
        }
        pair = {"mention_a": a, "mention_b": b}
        self.assertEqual(explicit_province(a), "tiền giang")
        self.assertEqual(explicit_province(b), "bến tre")
        self.assertTrue(has_explicit_province_conflict(pair))
        self.assertEqual(location_province_features(a, b)["conflicting_explicit_province"], 1.0)
        self.assertEqual(predict(pair, "exact_surface"), 0)

    def test_v3_case_455248_uses_the_marked_address_not_the_previous_party(self):
        a = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": (
                "Cư trú tại: Ấp Phú Bình, xã An Phước, huyện Châu Thành, tỉnh Bến Tre.\n"
                "- Bị đơn: Anh Huỳnh Văn Phước, sinh năm 1983;\n"
                "Cư trú tại: Ấp Minh Tân, xã Song Thuận, huyện [MENTION]Châu Thành[/MENTION], "
                "tỉnh Tiền Giang."
            ),
        }
        b = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": (
                "Nơi nhận:\n- TAND tỉnh Tiền Giang;\n- VKSND huyện Gò Công Tây;\n"
                "- UBND xã An Phước, huyện [MENTION]Châu Thành[/MENTION], tỉnh Bến Tre."
            ),
        }
        pair = {"mention_a": a, "mention_b": b}
        self.assertEqual(explicit_province(a), "tiền giang")
        self.assertEqual(explicit_province(b), "bến tre")
        self.assertTrue(has_explicit_province_conflict(pair))
        self.assertEqual(predict(pair, "exact_surface"), 0)

    def test_different_communes_in_same_province_are_not_vetoed(self):
        a = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": "xã An Phước, huyện [MENTION]Châu Thành[/MENTION], tỉnh Bến Tre.",
        }
        b = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": "xã Tân Phú, huyện [MENTION]Châu Thành[/MENTION], tỉnh Bến Tre.",
        }
        self.assertFalse(has_explicit_province_conflict({"mention_a": a, "mention_b": b}))
        self.assertEqual(location_province_features(a, b)["same_explicit_province"], 1.0)

    def test_location_veto_does_not_apply_to_people_or_ambiguous_context(self):
        person_a = {
            "label": "PER",
            "surface": "Nguyễn Văn An",
            "context": "Cư trú tỉnh [MENTION]Bến Tre[/MENTION].",
        }
        person_b = {
            "label": "PER",
            "surface": "Nguyễn Văn An",
            "context": "Cư trú tỉnh [MENTION]Tiền Giang[/MENTION].",
        }
        self.assertFalse(has_explicit_province_conflict({"mention_a": person_a, "mention_b": person_b}))
        ambiguous = {
            "label": "LOC",
            "surface": "Châu Thành",
            "context": "tỉnh Bến Tre và tỉnh Tiền Giang, huyện [MENTION]Châu Thành[/MENTION].",
        }
        self.assertIsNone(explicit_province(ambiguous))

    def test_province_extraction_stops_at_official_name(self):
        cases = {
            "[MENTION]Bình Xuyên[/MENTION], tỉnh Vĩnh Phúc có hiệu lực thi hành.": "vĩnh phúc",
            "[MENTION]Thanh Liêm[/MENTION], tỉnh Hà Nam theo biên bản.": "hà nam",
            "[MENTION]Bảo Lâm[/MENTION], tỉnh Cao Bằng do Tòa án xác minh.": "cao bằng",
            "[MENTION]Cầu Mới[/MENTION], TP. Hồ Chí Minh.": "hồ chí minh",
            "[MENTION]Cầu Mới[/MENTION], TP.HCM.": "hồ chí minh",
        }
        for context, expected in cases.items():
            with self.subTest(context=context):
                self.assertEqual(explicit_province({"context": context}), expected)

    def test_province_extraction_normalizes_dashes_and_vietnamese_diacritics(self):
        contexts = [
            "[MENTION]Mai Pha[/MENTION], tỉnh Bà Rịa - Vũng Tàu.",
            "[MENTION]Mai Pha[/MENTION], tỉnh Bà Rịa–Vũng Tàu.",
            "[MENTION]Mai Pha[/MENTION], tỉnh Ba Ria Vung Tau.",
        ]
        for context in contexts:
            with self.subTest(context=context):
                self.assertEqual(explicit_province({"context": context}), "bà rịa vũng tàu")

    def test_province_extraction_is_conservative_on_unknown_or_multiple_provinces(self):
        unknown = {"context": "[MENTION]Châu Thành[/MENTION], tỉnh Xứ Lạ."}
        ambiguous = {
            "context": "tỉnh Bến Tre và tỉnh Tiền Giang, huyện [MENTION]Châu Thành[/MENTION]."
        }
        false_word_boundary = {"context": "[MENTION]Châu Thành[/MENTION], tỉnh Bến Trex."}
        self.assertIsNone(explicit_province(unknown))
        self.assertIsNone(explicit_province(ambiguous))
        self.assertIsNone(explicit_province(false_word_boundary))

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

    def test_finetune_modes_and_automatic_lora_targets(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(4, 4)
                self.key = nn.Linear(4, 4)
                self.value = nn.Linear(4, 4)

        self.assertEqual(normalize_finetune_mode("fft", True), "frozen")
        self.assertEqual(normalize_finetune_mode("lora"), "lora")
        self.assertEqual(select_lora_targets(Encoder()), ["query", "key", "value"])
        with self.assertRaises(ValueError):
            normalize_finetune_mode("invalid")


if __name__ == "__main__":
    unittest.main()
