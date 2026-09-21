"""Tests for the court-personnel heuristic (core.roles)."""

import unittest

from vdt_anonymization.core.roles import classify_person_role, signature_block_start


class CourtPersonnelHeuristic(unittest.TestCase):
    DOCUMENT = (
        "TÒA ÁN NHÂN DÂN QUẬN NGŨ HÀNH SƠN\n"
        "- Thành phần Hội đồng xét xử sơ thẩm gồm có:\n"
        "Thẩm phán – Chủ tọa phiên tòa: Ông Trần Công Hoan\n"
        "Các hội thẩm nhân dân: Ông Trần Văn Sơn, Bà Huỳnh Thị Nga\n"
        "- Thư ký phiên tòa: Bà Vũ Thị Bích Hậu\n"
        "\nNỘI DUNG VỤ ÁN\n"
        "Nguyên đơn: Bà Nguyễn Thị An trình bày...\n"
        "Bị đơn: Ông Nguyễn Văn Bình không đồng ý...\n"
        "[1] Bà Nguyễn Thị An được nhận lại tiền tạm ứng án phí.\n"
        "\nNơi nhận:                         TM. HỘI ĐỒNG XÉT XỬ\n"
        "- Đương sự                        THẨM PHÁN- CHỦ TỌA PHIÊN TÒA\n"
        "- Lưu hồ sơ.                                    Trần Công Hoan\n"
    )

    def _role_of(self, name: str, occurrence: int = 0) -> str:
        start = -1
        for _ in range(occurrence + 1):
            start = self.DOCUMENT.index(name, start + 1)
        return classify_person_role(self.DOCUMENT, start, start + len(name))

    def test_header_composition_list_is_court_personnel(self):
        for name in ("Trần Công Hoan", "Trần Văn Sơn", "Huỳnh Thị Nga", "Vũ Thị Bích Hậu"):
            self.assertEqual(self._role_of(name), "court_personnel", name)

    def test_footer_signature_is_court_personnel(self):
        self.assertEqual(self._role_of("Trần Công Hoan", occurrence=1), "court_personnel")

    def test_declared_litigants_are_parties_not_court_personnel(self):
        self.assertEqual(self._role_of("Nguyễn Thị An"), "party")
        self.assertEqual(self._role_of("Nguyễn Văn Bình"), "party")

    def test_repeated_body_mention_of_a_litigant_stays_a_party(self):
        # A litigant mentioned again later in the body (not near any role
        # title, not in the footer) must not inherit court-personnel status
        # from the earlier, unrelated header block.
        self.assertEqual(self._role_of("Nguyễn Thị An", occurrence=1), "party")

    def test_footer_detection_uses_the_last_occurrence(self):
        # The composition list repeats "Thẩm phán - Chủ tọa phiên tòa"
        # language near the top; the footer must not be detected there.
        footer_start = signature_block_start(self.DOCUMENT)
        self.assertIsNotNone(footer_start)
        self.assertGreater(footer_start, self.DOCUMENT.index("NỘI DUNG VỤ ÁN"))

    def test_no_footer_present_does_not_crash(self):
        text = "Nguyên đơn: Bà Nguyễn Thị An trình bày..."
        self.assertIsNone(signature_block_start(text))
        self.assertEqual(classify_person_role(text, 20, 35), "party")


if __name__ == "__main__":
    unittest.main()
