import unittest

from vdt_anonymization.entity_linking.dataset import MENTION_CLOSE, MENTION_OPEN
from vdt_anonymization.entity_linking.clustering import complete_link_clusters
from vdt_anonymization.entity_linking.baseline import predict as baseline_predict
from vdt_anonymization.entity_linking.location_constraints import (
    explicit_province,
    location_pair_features,
    location_rule_decision,
    location_signature,
)


def loc(surface, before="", after="", label="LOC"):
    return {
        "label": label,
        "surface": surface,
        "context": f"{before}{MENTION_OPEN}{surface}{MENTION_CLOSE}{after}",
    }


class VietnameseLocationHierarchyTests(unittest.TestCase):
    def test_same_district_name_is_resolved_by_province_parent(self):
        ben_tre = loc("Châu Thành", "Địa chỉ: huyện ", ", tỉnh Bến Tre.")
        tien_giang = loc("Châu Thành", "Địa chỉ: huyện ", ", tỉnh Tiền Giang.")

        self.assertEqual(explicit_province(ben_tre), "bến tre")
        self.assertEqual(explicit_province(tien_giang), "tiền giang")
        self.assertEqual(location_rule_decision({"mention_a": ben_tre, "mention_b": tien_giang}), "block")

    def test_same_anonymized_location_alias_is_blocked_by_conflicting_provinces(self):
        ben_tre = loc("G", "Địa chỉ: huyện ", ", tỉnh Bến Tre.")
        tien_giang = loc("G", "Địa chỉ: huyện ", ", tỉnh Tiền Giang.")

        self.assertEqual(location_signature(ben_tre)["path"]["province"], {"83"})
        self.assertEqual(location_signature(tien_giang)["path"]["province"], {"82"})
        self.assertEqual(location_rule_decision({"mention_a": ben_tre, "mention_b": tien_giang}), "block")

    def test_same_commune_and_same_canonical_parent_path_rule_links(self):
        first = loc("An Phước", "Địa chỉ: xã ", ", huyện Châu Thành, tỉnh Bến Tre.")
        second = loc("An Phước", "Cư trú: xã ", ", huyện Châu Thành, tỉnh Bến Tre.")

        self.assertEqual(location_rule_decision({"mention_a": first, "mention_b": second}), "link")
        self.assertEqual(location_pair_features(first, second)["same_admin_unit"], 1.0)

    def test_gazetteer_unit_distinguishes_thanh_pho_district_from_province(self):
        district_city = loc("Bắc Ninh", "thành phố ", ", tỉnh Bắc Ninh.")
        province = loc("Bắc Ninh", "tỉnh ", ".")

        self.assertEqual(location_signature(district_city)["target"]["level"], "district")
        self.assertEqual(location_signature(province)["target"]["level"], "province")
        self.assertEqual(
            location_rule_decision({"mention_a": district_city, "mention_b": province}),
            "block",
        )

    def test_unique_ocr_typo_can_resolve_with_known_parent(self):
        exact = loc("An Phước", "Địa chỉ: xã ", ", huyện Châu Thành, tỉnh Bến Tre.")
        typo = loc("An Phuooc", "Cư trú: xã ", ", huyện Châu Thành, tỉnh Bến Tre.")

        signature = location_signature(typo)
        self.assertEqual(location_rule_decision({"mention_a": exact, "mention_b": typo}), "link")
        self.assertTrue(signature["target"]["fuzzy"])
        self.assertEqual(location_pair_features(exact, typo)["fuzzy_admin_match"], 1.0)
        self.assertEqual(
            baseline_predict({"mention_a": exact, "mention_b": typo}, "exact_surface"), 1
        )

    def test_low_level_alias_needs_shared_commune_or_district_parent(self):
        first = loc("Minh Tân", "Địa chỉ: ấp ", ", xã An Phước, huyện Châu Thành, tỉnh Bến Tre.")
        second = loc("Minh Tân", "Cư trú: ấp ", ", xã An Phước, huyện Châu Thành, tỉnh Bến Tre.")
        no_parent = loc("Minh Tân", "Địa chỉ: ấp ", ", tỉnh Bến Tre.")
        district_only = loc("Minh Tân", "Địa chỉ: ấp ", ", huyện Châu Thành, tỉnh Bến Tre.")

        self.assertEqual(location_rule_decision({"mention_a": first, "mention_b": second}), "link")
        self.assertIsNone(location_rule_decision({"mention_a": first, "mention_b": no_parent}))
        self.assertIsNone(location_rule_decision({"mention_a": district_only, "mention_b": district_only}))

    def test_different_low_level_unit_types_are_not_auto_linked(self):
        hamlet = loc("Minh Tân", "ấp ", ", xã An Phước, huyện Châu Thành, tỉnh Bến Tre.")
        village = loc("Minh Tân", "xóm ", ", xã An Phước, huyện Châu Thành, tỉnh Bến Tre.")

        self.assertIsNone(location_rule_decision({"mention_a": hamlet, "mention_b": village}))

    def test_conflicting_child_communes_do_not_block_same_district_target(self):
        first = loc(
            "L", "Địa chỉ: thị xã ", ", tỉnh Bình Thuận, xã Tân Bình."
        )
        second = loc(
            "L", "Địa chỉ: thị xã ", ", tỉnh Bình Thuận, xã Tân Tiến."
        )

        self.assertEqual(location_signature(first)["target"]["level"], "district")
        self.assertEqual(location_signature(second)["target"]["level"], "district")
        self.assertEqual(
            location_signature(first)["target"]["code"],
            location_signature(second)["target"]["code"],
        )
        self.assertEqual(location_rule_decision({"mention_a": first, "mention_b": second}), "link")

    def test_ocr_split_unit_and_single_line_wrap_are_repaired(self):
        district = loc("T2", "UBND xã T, hu yện ", ", tỉnh Thanh Hóa.")
        wrapped_city = loc("T", "thành ph ố \n", ", tỉnh Phú Thọ.")

        self.assertEqual(location_signature(district)["target"]["level"], "district")
        self.assertEqual(location_signature(wrapped_city)["target"]["level"], "ambiguous")
        self.assertIsNone(location_signature(wrapped_city)["target"]["code"])

    def test_short_anonymization_alias_is_not_globally_resolved_as_a_real_place(self):
        signature = location_signature(loc("V"))

        self.assertIsNone(signature["target"])

    def test_ner_org_type_is_never_overridden_by_location_words(self):
        organization = loc(
            "Tòa án nhân dân huyện Châu Thành",
            "",
            ", tỉnh Bến Tre.",
            label="ORG",
        )
        location = loc("Châu Thành", "Địa chỉ: huyện ", ", tỉnh Bến Tre.")

        self.assertEqual(location_signature(organization), {})
        self.assertIsNone(location_rule_decision({"mention_a": organization, "mention_b": location}))
        self.assertEqual(location_pair_features(organization, location)["same_admin_unit"], 0.0)

    def test_rule_link_precedes_model_threshold_but_conflict_still_wins(self):
        mentions = [{"label": "LOC", "start": 0}, {"label": "LOC", "start": 10}]
        self.assertEqual(
            complete_link_clusters(mentions, {(0, 1): 0.01}, 0.9, must_link={(0, 1)}),
            [[0, 1]],
        )
        self.assertEqual(
            complete_link_clusters(
                mentions, {(0, 1): 0.99}, 0.5,
                cannot_link={(0, 1)}, must_link={(0, 1)},
            ),
            [[0], [1]],
        )


if __name__ == "__main__":
    unittest.main()
