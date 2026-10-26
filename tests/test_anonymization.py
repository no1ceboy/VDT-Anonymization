import unittest
from vdt_anonymization.entity_linking.dataset import MENTION_CLOSE, MENTION_OPEN
from vdt_anonymization.core.anonymization import (
    apply_span_replacements,
    build_anonymization_plan,
)


def person_cluster(text, entity_id, surfaces):
    mentions = []
    cursor = 0
    for surface in surfaces:
        start = text.index(surface, cursor)
        end = start + len(surface)
        mentions.append({"label": "PER", "start": start, "end": end, "surface": surface})
        cursor = end
    return {"cluster_id": entity_id, "label": "PER", "mentions": mentions}


def mention_at(text, surface, start):
    return {"label": "PER", "start": start, "end": start + len(surface), "surface": surface}


def contextual_mention(text, surface, start, label="LOC"):
    end = start + len(surface)
    left, right = max(0, start - 180), min(len(text), end + 180)
    context = (text[left:start] + MENTION_OPEN + text[start:end]
               + MENTION_CLOSE + text[end:right])
    return {"label": label, "start": start, "end": end,
            "surface": surface, "context": context}


def org_cluster(text, entity_id, surfaces):
    mentions = []
    cursor = 0
    for surface in surfaces:
        start = text.index(surface, cursor)
        end = start + len(surface)
        mentions.append({"label": "ORG", "start": start, "end": end, "surface": surface})
        cursor = end
    return {"cluster_id": entity_id, "label": "ORG", "mentions": mentions}


class AnonymizationTests(unittest.TestCase):
    def test_many_distinct_full_names_sharing_initial_are_not_numbered(self):
        names = [
            "Nguyễn Văn Tâm", "Trần Thị Tình", "Lê Văn Toàn", "Phạm Thị Thoa",
            "Hoàng Văn Tài", "Võ Thị Tuyết", "Đặng Văn Tùng", "Bùi Thị Thủy",
            "Đỗ Văn Tiến", "Hồ Thị Thanh", "Ngô Văn Thành", "Dương Thị Tâm",
            "Lý Văn Trọng", "Mai Thị Trang",
        ]
        text = "; ".join(names)
        clusters = [person_cluster(text, f"P{i:02d}", [name]) for i, name in enumerate(names)]

        plan = build_anonymization_plan(text, clusters)

        codes = {entity["code"] for entity in plan["entities"]}
        self.assertEqual(codes, {"T"})
        self.assertEqual(len(plan["entities"]), 14)
        self.assertFalse(any(entity["code"] == "T14" for entity in plan["entities"]))

    def test_one_person_uses_same_code_for_full_and_short_name(self):
        text = "Nguyễn Thị Anh trình bày; sau đó Anh bổ sung ý kiến."
        cluster = person_cluster(text, "P1", ["Nguyễn Thị Anh", "Anh"])

        plan = build_anonymization_plan(text, [cluster])
        output = apply_span_replacements(text, plan["replacements"])

        self.assertEqual(plan["entities"][0]["code"], "A")
        self.assertEqual(output, "Nguyễn Thị A trình bày; sau đó A bổ sung ý kiến.")

    def test_distinct_people_with_same_short_alias_get_distinct_stable_codes(self):
        text = "Nguyễn Văn An trình bày. Trần Thị An bổ sung. An tiếp tục khai. An phản hồi."
        first_alias = text.index("An tiếp tục")
        second_alias = text.index("An phản hồi")
        clusters = [
            {"cluster_id": "P1", "label": "PER", "mentions": [
                mention_at(text, "Nguyễn Văn An", text.index("Nguyễn Văn An")),
                mention_at(text, "An", first_alias),
            ]},
            {"cluster_id": "P2", "label": "PER", "mentions": [
                mention_at(text, "Trần Thị An", text.index("Trần Thị An")),
                mention_at(text, "An", second_alias),
            ]},
        ]

        plan = build_anonymization_plan(text, clusters)
        output = apply_span_replacements(text, plan["replacements"])

        codes = {entity["entity_id"]: entity["code"] for entity in plan["entities"]}
        self.assertEqual(codes, {"P1": "A1", "P2": "A2"})
        self.assertEqual(
            output,
            "Nguyễn Văn A1 trình bày. Trần Thị A2 bổ sung. A1 tiếp tục khai. A2 phản hồi.",
        )

    def test_organization_codes_are_document_local_not_name_initials(self):
        text = "Công ty TNHH An Gia ký hợp đồng. Công ty TNHH An Phát phản hồi. An Gia nộp đơn."
        first = org_cluster(text, "O1", ["Công ty TNHH An Gia", "An Gia"])
        second = org_cluster(text, "O2", ["Công ty TNHH An Phát"])

        plan = build_anonymization_plan(text, [first, second])
        output = apply_span_replacements(text, plan["replacements"])

        codes = {entity["entity_id"]: entity["code"] for entity in plan["entities"]}
        self.assertEqual(codes, {"O1": "B", "O2": "C"})
        self.assertEqual(
            output,
            "Công ty TNHH B ký hợp đồng. Công ty TNHH C phản hồi. B nộp đơn.",
        )

    def test_organization_codes_do_not_shift_person_collision_numbers(self):
        text = (
            "Công ty TNHH Trường An; Công ty TNHH Tân Phát. "
            "Nguyễn Văn T, T khai. Trần Văn T, T phản hồi."
        )
        first_alias = text.index("T khai")
        second_alias = text.index("T phản hồi")
        organizations = [
            org_cluster(text, "O1", ["Công ty TNHH Trường An"]),
            org_cluster(text, "O2", ["Công ty TNHH Tân Phát"]),
        ]
        people = [
            {"cluster_id": "P1", "label": "PER", "mentions": [
                mention_at(text, "Nguyễn Văn T", text.index("Nguyễn Văn T")),
                mention_at(text, "T", first_alias),
            ]},
            {"cluster_id": "P2", "label": "PER", "mentions": [
                mention_at(text, "Trần Văn T", text.index("Trần Văn T")),
                mention_at(text, "T", second_alias),
            ]},
        ]

        plan = build_anonymization_plan(text, organizations + people)

        codes = {entity["entity_id"]: entity["code"] for entity in plan["entities"]}
        self.assertEqual(codes, {"O1": "B", "O2": "C", "P1": "T1", "P2": "T2"})

    def test_embedded_ubnd_locality_matches_address_location_with_real_parser(self):
        text = (
            "Ủy ban nhân dân thị xã Điện Bàn, tỉnh Quảng Nam. "
            "Địa chỉ: đường X, phường Y, thị xã Điện Bàn, tỉnh Quảng Nam."
        )
        org_surface = "Ủy ban nhân dân thị xã Điện Bàn, tỉnh Quảng Nam"
        inner_org_surface = "Ủy ban nhân"
        nested_loc_surface = "Điện Bàn"
        standalone_loc_surface = "thị xã Điện Bàn"
        org_start = text.index(org_surface)
        inner_org_start = text.index(inner_org_surface)
        nested_loc_start = text.index(nested_loc_surface)
        standalone_loc_start = text.rindex(standalone_loc_surface)
        loc_clusters = [{
            "cluster_id": "L1",
            "label": "LOC",
            "mentions": [
                contextual_mention(text, nested_loc_surface, nested_loc_start),
                contextual_mention(text, standalone_loc_surface, standalone_loc_start),
            ],
        }]
        org_clusters = [
            {"cluster_id": "O1", "label": "ORG", "mentions": [
                {"label": "ORG", "start": org_start,
                 "end": org_start + len(org_surface), "surface": org_surface},
            ]},
            {"cluster_id": "O2", "label": "ORG", "mentions": [
                {"label": "ORG", "start": inner_org_start,
                 "end": inner_org_start + len(inner_org_surface), "surface": inner_org_surface},
            ]},
        ]

        plan = build_anonymization_plan(text, org_clusters + loc_clusters)

        output = apply_span_replacements(text, plan["replacements"])
        self.assertEqual(plan["status"], "ready")
        self.assertIn("Ủy ban nhân dân thị xã", output)
        self.assertIn(", tỉnh Quảng Nam", output)
        self.assertNotIn("Điện Bàn", output)
        location_codes = [
            item["replacement"].split()[-1]
            for item in plan["replacements"]
            if item["label"] == "LOC" and item["start"] == standalone_loc_start
        ]
        org_replacement = next(
            item["replacement"] for item in plan["replacements"] if item["label"] == "ORG"
        )
        self.assertEqual(location_codes, [org_replacement.split("thị xã ", 1)[1].split(",", 1)[0]])
        self.assertFalse(plan["conflicting_entity_ids"])
        self.assertEqual(
            sum(item["label"] == "ORG" for item in plan["replacements"]), 1
        )


if __name__ == "__main__":
    unittest.main()
