"""Build a synthetic un-anonymized legal dataset from NER predictions.

This is not identity recovery. It creates plausible, deterministic fake
identities for anonymized mentions in already published legal documents.

The pipeline is deliberately recall-oriented:

1. Read PER/ORG predictions and optional LOC fallback predictions from ``run_ner.py``.
2. Scan the complete source text for court-style markers such as ``M``, ``L``
   and ``L1`` that NER may miss.
3. Apply structured address rules before using NER labels.
4. Classify marker candidates using nearby person/organization/location structure.
5. Link identical markers within a document and semantic role.
6. Generate one synthetic value per linked entity.
7. Replace linked mentions from right to left.

No network, model, or third-party package is required. The NER JSONL may be
missing or incomplete; marker recovery still runs over the source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict


DEFAULT_SOURCE = "datasets/legal_test.jsonl"
DEFAULT_NER = "outputs/nlphust_legal_test.jsonl"
DEFAULT_DATASET_OUTPUT = "outputs/synthetic_unanonymized.jsonl"
DEFAULT_LINK_OUTPUT = "outputs/entity_links.jsonl"
DEFAULT_MAP_OUTPUT = "outputs/replacement_maps.jsonl"

WORD_RE = re.compile(r"(?u)[^\W\d_]+(?:\d{1,2})?")
# Normal spaced marker: L, M, L1, H2, ... . Two-digit evidence codes such as
# A01/A03 are intentionally excluded because they are not person markers.
MARKER_RE = re.compile(r"(?<![\w])([A-ZĐ](?:\d)?)(?![\w])")
# OCR/text-extraction variant: ĐứcH, VănL1.
COMPACT_MARKER_RE = re.compile(r"(?<=[^\W\d_])([A-ZĐ](?:\d)?)(?![\w])")

PERSON_CUE_RE = re.compile(
    r"(?<!\w)(?:ông|bà|anh|chị|em|cô|chú|bác|con|vợ|chồng|cha|mẹ|"
    r"bị\s+cáo|nguyên\s+đơn|bị\s+đơn|người\s+bị\s+hại|"
    r"người\s+có\s+quyền\s+lợi|người\s+liên\s+quan|người\s+làm\s+chứng|"
    r"luật\s+sư|đại\s+diện|đương\s+sự|bị\s+can|người\s+khởi\s+kiện)(?!\w)",
    re.IGNORECASE,
)
ORGANIZATION_CUE_RE = re.compile(
    r"(?<!\w)(?:ngân\s+hàng|công\s+ty|tập\s+đoàn|doanh\s+nghiệp|chi\s+nhánh|"
    r"văn\s+phòng|hợp\s+tác\s+xã|trường|viện|ủy\s+ban|toà\s+án|tòa\s+án|"
    r"bộ|sở|cục|học\s+viện|quỹ|ban\s+quản\s+lý)(?!\w)",
    re.IGNORECASE,
)
LOCATION_CUE_RE = re.compile(
    r"(?<!\w)(?:xã|phường|thị\s+trấn|huyện|quận|thị\s+xã|thành\s+phố|tỉnh|"
    r"ấp|thôn|khu\s+phố|đường|ngõ|hẻm|khu\s+dân\s+cư|tòa\s+án|tand)(?!\w)",
    re.IGNORECASE,
)
ADDRESS_SLOT_PATTERNS = (
    ("house_number", re.compile(r"(?:\bsố(?:\s+nhà)?)\s*$", re.IGNORECASE)),
    ("floor", re.compile(r"(?:\btầng)\s*$", re.IGNORECASE)),
    ("room", re.compile(r"(?:\bphòng|\bcăn)\s*$", re.IGNORECASE)),
    ("parcel", re.compile(r"(?:\blô|\bthửa|\btờ)\s*$", re.IGNORECASE)),
    ("hamlet", re.compile(r"(?:\bấp|\bthôn|\bkhu\s+phố|\btổ\s+dân\s+phố|\bkhu\s+dân\s+cư)\s*$", re.IGNORECASE)),
    ("commune", re.compile(r"(?:\bxã|\bphường|\bthị\s+trấn)\s*$", re.IGNORECASE)),
    ("district", re.compile(r"(?:\bhuyện|\bquận|\bthị\s+xã|\bthành\s+phố)\s*$", re.IGNORECASE)),
    ("province", re.compile(r"(?:\btỉnh|\bthành\s+phố)\s*$", re.IGNORECASE)),
    ("street", re.compile(r"(?:\bđường|\bngõ|\bhẻm)\s*$", re.IGNORECASE)),
)
TITLE_RE = re.compile(
    r"^(?:ông|bà|anh|chị|em|cô|chú|bác|ông/bà|bà/ông)\s+",
    re.IGNORECASE,
)
FEMALE_GENDER_CUE_RE = re.compile(r"\b(?:bà|chị|cô|mẹ|vợ|nữ|thị)\b", re.IGNORECASE)
MALE_GENDER_CUE_RE = re.compile(r"\b(?:ông|anh|chú|cha|chồng|nam|văn)\b", re.IGNORECASE)
LOCATION_PREFIX_RE = re.compile(
    r"^(?:xã|phường|thị\s+trấn|huyện|quận|thị\s+xã|thành\s+phố|tỉnh|"
    r"ấp|thôn|khu\s+phố|đường|ngõ|hẻm)\s+",
    re.IGNORECASE,
)
ADMINISTRATIVE_PREFIX_RE = re.compile(
    r"(?:xã|phường|thị\s+trấn|huyện|quận|thị\s+xã|thành\s+phố|tỉnh|"
    r"ấp|thôn|khu\s+phố|đường|ngõ|hẻm)\s*$",
    re.IGNORECASE,
)
PERSON_MARKER_PREFIX_RE = re.compile(
    r"(?:ông|bà|anh|chị|em|cô|chú|bác|ông/bà|bà/ông|bị\s+cáo|"
    r"nguyên\s+đơn|bị\s+đơn|người\s+bị\s+hại|văn|thị)\s*$",
    re.IGNORECASE,
)

ENTITY_ALIASES = {
    "PER": "PER",
    "PERSON": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "ADDR": "ADDR",
    "ADDRESS": "ADDR",
}

# These are synthetic organization-name components, not a list of real
# companies.  Their Cartesian product provides enough deterministic variety
# for large datasets while keeping names pronounceable and common in Vietnam.
ORGANIZATION_NAME_LEFT = [
    "An", "Bình", "Cao", "Đại", "Đông", "Gia", "Hải", "Hòa", "Hưng", "Kim",
    "Long", "Minh", "Nam", "Phú", "Quang", "Sơn", "Tân", "Thái", "Thanh",
    "Thiên", "Thịnh", "Trường", "Việt", "Vĩnh", "Xuân",
]
ORGANIZATION_NAME_RIGHT = [
    "An", "Bình", "Châu", "Đức", "Gia", "Hải", "Hòa", "Khang", "Long", "Minh",
    "Phát", "Phú", "Quang", "Sơn", "Tâm", "Thành", "Thịnh", "Tiến", "Trung",
    "Việt", "Vinh", "Yên", "Nguyên", "Đạt", "Lộc",
]
SYNTHETIC_ORGANIZATION_NAMES = list(dict.fromkeys([
    "Minh Việt", "Đại Việt", "An Phát", "Hưng Thịnh", "Tân Thành", "Phú Gia",
    "Thịnh Vượng", "Thanh Bình", "Hoàng Gia", "Việt An",
    *[
    f"{left} {right}"
    for left in ORGANIZATION_NAME_LEFT
    for right in ORGANIZATION_NAME_RIGHT
    if left != right
    ],
]))

# These are given-name pools, not recovered identities. They contain only
# common, single-token Vietnamese given names. We preserve the anonymized
# initial when a suitable common name exists, but do not force an uncommon or
# invented name just to match an arbitrary publication marker.
GIVEN_NAMES = {
    "male": {
        "A": ["An", "Anh"],
        "B": ["Bảo", "Bình"],
        "C": ["Cường", "Chính", "Công"],
        "D": ["Dũng", "Duy", "Dương"],
        "Đ": ["Đạt", "Đức", "Đăng"],
        "G": ["Gia", "Giang"],
        "H": ["Hải", "Hùng", "Hoàng", "Hưng", "Hiếu", "Huy"],
        "K": ["Khoa", "Kiên", "Khôi", "Khánh", "Khang"],
        "L": ["Long", "Lâm", "Lộc"],
        "M": ["Minh", "Mạnh"],
        "N": ["Nam", "Nghĩa", "Nhân", "Nguyên", "Ngọc"],
        "P": ["Phúc", "Phong", "Phát", "Phước"],
        "Q": ["Quang", "Quốc", "Quân"],
        "S": ["Sơn", "Sang", "Sinh"],
        "T": ["Tuấn", "Tùng", "Trung", "Thành", "Thắng", "Tiến", "Tâm", "Thịnh", "Trí"],
        "V": ["Việt", "Vinh", "Vũ", "Vương"],
        "X": ["Xuân"],
    },
    "female": {
        "A": ["Ánh", "An", "Anh"],
        "B": ["Bích", "Bình", "Bảo"],
        "C": ["Chi", "Châu", "Cúc"],
        "D": ["Diễm", "Dung", "Duyên", "Diệp"],
        "Đ": ["Đan", "Đào"],
        "G": ["Giang", "Giao"],
        "H": ["Hạnh", "Hoa", "Hương", "Hiền", "Hồng", "Hà", "Hằng", "Huyền", "Hoài"],
        "K": ["Kim", "Kiều", "Khánh"],
        "L": ["Lan", "Linh", "Loan", "Liên"],
        "M": ["Mai", "My", "Mỹ"],
        "N": ["Ngân", "Nga", "Ngọc", "Nhung", "Nhi", "Như", "Nguyên"],
        "P": ["Phương", "Phượng"],
        "Q": ["Quỳnh", "Quyên"],
        "S": ["Sen", "Sương"],
        "T": ["Trang", "Thảo", "Trinh", "Thúy", "Thanh", "Tâm", "Tiên", "Tuyết", "Tú"],
        "V": ["Vân", "Vy", "Vi"],
        "X": ["Xuân"],
        "Y": ["Yến"],
        "U": ["Uyên"],
        "O": ["Oanh"],
    },
}

FAMILY_NAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Võ", "Vũ", "Đặng", "Bùi",
    "Phan", "Đoàn", "Đinh", "Dương", "Hồ", "Ngô", "Đỗ", "Tô", "Cao", "Lý", "Lưu",
    "Trương", "Mai", "Mạc", "Hà", "Giang", "Chung", "Chu", "Thái", "Vương", "Lương",
]

FALLBACK_GIVEN_NAMES = {
    "male": ["Anh", "Bình", "Dũng", "Hải", "Hùng", "Khoa", "Long", "Minh", "Nam", "Phúc", "Quang", "Sơn", "Tuấn", "Vinh"],
    "female": ["Anh", "Bình", "Dung", "Hạnh", "Hoa", "Hương", "Lan", "Linh", "Mai", "Nga", "Ngọc", "Phương", "Trang", "Thảo", "Vy"],
}

# Only these tokens may be used to expand a one-letter marker into a full
# name. This prevents words such as ``Bên``, ``Cháu`` or all-caps headings from
# becoming fake name prefixes when the marker regex scans the whole document.
NAME_PREFIX_TOKENS = {
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "võ", "vũ", "đặng", "bùi",
    "phan", "đoàn", "đinh", "dương", "hồ", "ngô", "đỗ", "tô", "tạ", "cao", "mai",
    "lý", "lưu", "trương", "mạc", "mã", "hứa", "giàng", "chung", "chu", "hà", "hạ",
    "lâm", "tống", "tôn", "kiều", "khổng", "thân", "bạch", "vi", "thái", "quách",
    "triệu", "tăng", "từ", "nông", "sầm", "thạch", "vương", "lương", "uông", "nghiêm",
    "lại", "chế", "văn", "thị", "đức", "hữu", "quốc", "ngọc", "minh", "thanh", "đình",
    "công", "thế", "xuân", "kim", "thúy", "nhật", "quang", "tấn", "thành", "anh", "bảo",
    "gia", "khắc", "trọng", "phước", "phú", "cẩm", "hải", "mạnh", "đình", "văn",
    "trúc", "bích", "bạch", "hoài", "thùy", "tuệ", "thái", "khắc", "tú", "tường",
    "huyền", "diệp", "phương", "phượng", "quỳnh", "quyên", "lan", "linh", "loan",
}

# Plausible synthetic place names by administrative unit. These names are
# intentionally generic and are selected deterministically by marker/index.
LOCATION_NAMES = {
    "province": {
        "A": ["An Bình"], "B": ["Bình An"], "C": ["Cao Sơn"],
        "D": ["Đông Hải"], "Đ": ["Đông Hải"], "H": ["Hòa Bình"],
        "L": ["Lâm An"], "M": ["Minh Hải"], "N": ["Nam Sơn"],
        "P": ["Phú An"], "Q": ["Quang Minh"], "S": ["Sơn Hà"],
        "T": ["Tân Bình"], "V": ["Vĩnh An"],
    },
    "district": {
        "A": ["An Sơn"], "B": ["Bình Sơn"], "C": ["Cao Sơn"],
        "D": ["Đông Sơn"], "Đ": ["Đức Sơn"], "H": ["Hòa Sơn"],
        "L": ["Lộc Bình"], "M": ["Minh Sơn"], "N": ["Nam Sơn"],
        "P": ["Phú Sơn"], "Q": ["Quang Sơn"], "S": ["Sơn Tây"],
        "T": ["Tân Sơn"], "V": ["Vĩnh Sơn"],
    },
    "commune": {
        "A": ["An Hòa"], "B": ["Bình Minh"], "C": ["Cao An"],
        "D": ["Đông An"], "Đ": ["Đức An"], "H": ["Hòa An"],
        "L": ["Long An"], "M": ["Minh An"], "N": ["Nam An"],
        "P": ["Phú An"], "Q": ["Quang An"], "S": ["Sơn An"],
        "T": ["Tân An"], "V": ["Vĩnh An"],
    },
    "hamlet": {
        "A": ["An Bình"], "B": ["Bình An"], "C": ["Cầu Mới"],
        "D": ["Đông Bình"], "Đ": ["Đức Hòa"], "H": ["Hòa Bình"],
        "L": ["Long Bình"], "M": ["Minh Tân"], "N": ["Nam Bình"],
        "P": ["Phú Bình"], "Q": ["Quang Trung"], "S": ["Sơn Bình"],
        "T": ["Tân Bình"], "V": ["Vĩnh Bình"],
    },
    "generic": {
        "A": ["An Bình"], "B": ["Bình Minh"], "C": ["Cao Sơn"],
        "D": ["Đông An"], "Đ": ["Đức An"], "H": ["Hòa Bình"],
        "L": ["Long An"], "M": ["Minh An"], "N": ["Nam Sơn"],
        "P": ["Phú An"], "Q": ["Quang Minh"], "S": ["Sơn Hà"],
        "T": ["Tân Bình"], "V": ["Vĩnh An"],
    },
}


def canonical_label(value):
    label = str(value or "").strip().upper()
    if "-" in label:
        label = label.split("-", 1)[-1]
    return ENTITY_ALIASES.get(label, label)


def normalize_for_match(value):
    value = unicodedata.normalize("NFC", str(value or "")).casefold()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def marker_value(value):
    value = str(value or "").strip().upper()
    return value if re.fullmatch(r"[A-ZĐ](?:\d)?", value) else None


def is_name_prefix_token(token):
    return normalize_for_match(token) in NAME_PREFIX_TOKENS


def marker_initial(marker):
    return marker[0].upper()


def initial_key(value):
    if not value:
        return ""
    first = unicodedata.normalize("NFD", str(value).strip()[0])
    return "".join(character for character in first if unicodedata.category(character) != "Mn").upper()


def marker_index(marker):
    suffix = marker[1:]
    return int(suffix) if suffix else 0


def load_jsonl_by_id(path):
    records = {}
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", row.get("case_id", row.get("id", line_number))))
            records[doc_id] = row
    return records


def iter_source_rows(path, offset=0, limit=0):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle):
            if row_number < offset:
                continue
            if limit > 0 and row_number >= offset + limit:
                break
            if line.strip():
                yield row_number, json.loads(line)


def context_scores(text, start, end, window=100):
    left = max(0, start - window)
    right = min(len(text), end + window)
    context = text[left:right]
    return len(PERSON_CUE_RE.findall(context)), len(LOCATION_CUE_RE.findall(context))


def nearest_cue_distance(text, start, end, pattern, window=100):
    left = max(0, start - window)
    right = min(len(text), end + window)
    distances = []
    for match in pattern.finditer(text[left:right]):
        cue_start = left + match.start()
        cue_end = left + match.end()
        if cue_end <= start:
            distances.append(start - cue_end)
        elif cue_start >= end:
            distances.append(cue_start - end)
        else:
            distances.append(0)
    return min(distances) if distances else None


def preceding_cue_distance(text, start, pattern, window=100):
    """Distance from a marker to the nearest cue occurring before it."""
    left = max(0, start - window)
    distances = []
    for match in pattern.finditer(text[left:start]):
        distances.append(start - (left + match.end()))
    return min(distances) if distances else None


def structured_address_role(text, start, end, window=80):
    """Return the address slot immediately governing a value.

    Legal addresses have a strong local grammar.  Looking only at the text
    immediately before the marker avoids confusing a nearby address cue in a
    different clause with the marker being classified.
    """
    before = text[max(0, start - window):start]
    for role, pattern in ADDRESS_SLOT_PATTERNS:
        if pattern.search(before):
            return role

    # NER often returns the cue together with its value (``Ấp X`` or
    # ``Số X``), so inspect the beginning of the predicted surface too.
    surface = text[start:end]
    surface_prefixes = (
        ("house_number", re.compile(r"^\s*số(?:\s+nhà)?\s+", re.IGNORECASE)),
        ("floor", re.compile(r"^\s*tầng\s+", re.IGNORECASE)),
        ("room", re.compile(r"^\s*(?:phòng|căn)\s+", re.IGNORECASE)),
        ("parcel", re.compile(r"^\s*(?:lô|thửa|tờ)\s+", re.IGNORECASE)),
        ("hamlet", re.compile(r"^\s*(?:ấp|thôn|khu\s+phố|tổ\s+dân\s+phố|khu\s+dân\s+cư)\s+", re.IGNORECASE)),
        ("commune", re.compile(r"^\s*(?:xã|phường|thị\s+trấn)\s+", re.IGNORECASE)),
        ("district", re.compile(r"^\s*(?:huyện|quận|thị\s+xã|thành\s+phố)\s+", re.IGNORECASE)),
        ("province", re.compile(r"^\s*(?:tỉnh|thành\s+phố)\s+", re.IGNORECASE)),
        ("street", re.compile(r"^\s*(?:đường|ngõ|hẻm)\s+", re.IGNORECASE)),
    )
    for role, pattern in surface_prefixes:
        if pattern.search(surface):
            return role
    return None


def mention_role(text, mention):
    """Return a stable semantic role used to link repeated markers."""
    label = canonical_label(mention.get("label"))
    if label == "ADDR":
        return structured_address_role(text, mention["start"], mention["end"]) or "address"
    if label == "LOC":
        slot = structured_address_role(text, mention["start"], mention["end"])
        return f"location:{slot}" if slot else "location:freeform"
    if label == "ORG":
        return "organization"
    if label == "PER":
        return "person"
    return label.lower()


def classify_marker(text, start, end, ner_label=None):
    label = canonical_label(ner_label)

    person_distance = nearest_cue_distance(text, start, end, PERSON_CUE_RE)
    organization_distance = nearest_cue_distance(text, start, end, ORGANIZATION_CUE_RE)
    person_before_distance = preceding_cue_distance(text, start, PERSON_CUE_RE)
    organization_before_distance = preceding_cue_distance(text, start, ORGANIZATION_CUE_RE)
    location_distance = nearest_cue_distance(text, start, end, LOCATION_CUE_RE)
    max_cue_distance = 24
    max_organization_distance = 60

    # Address structure is the highest-confidence signal.  In particular,
    # ``Số X`` is an address number, not a location entity, while ``Ấp X`` is
    # a locality.  This rule also overrides an incorrect LOC NER prediction.
    address_role = structured_address_role(text, start, end)
    if address_role in {"house_number", "floor", "room", "parcel"}:
        return "ADDR", "address_structure"
    if address_role:
        return "LOC", f"address_structure:{address_role}"

    # Organization cues are more reliable than a broad nearby-location window
    # for forms such as ``Ngân hàng thương mại cổ phần X``.  A closer person
    # cue still wins for a person mentioned in an organization-related clause.
    person_is_closer = (
        person_before_distance is not None
        and person_before_distance <= max_cue_distance
        and organization_before_distance is not None
        and person_before_distance < organization_before_distance
    )
    if (
        organization_before_distance is not None
        and organization_before_distance <= max_organization_distance
        and not person_is_closer
    ):
        return "ORG", "organization_context"

    surface = text[start:end]
    marker = marker_in_span(text, start, end)
    prefix = ""
    if marker:
        surface_marker_matches = list(MARKER_RE.finditer(surface))
        if not surface_marker_matches:
            surface_marker_matches = [
                match for match in COMPACT_MARKER_RE.finditer(surface)
                if match.start(1) > 0 and surface[match.start(1) - 1].islower()
            ]
        if surface_marker_matches:
            prefix = surface[:surface_marker_matches[-1].start(1)].strip()
    prefix_tokens = prefix.split()
    has_name_shape = bool(prefix_tokens) and any(
        is_name_prefix_token(token) for token in prefix_tokens
    ) and (
        len(prefix_tokens) >= 2
        or any(normalize_for_match(token) in {"thị", "văn"} for token in prefix_tokens)
    )

    # A recovered form such as Nguyễn Thị L1 or Đặng ĐứcH is stronger than a
    # nearby unrelated location cue. The prefix is retained in the public
    # text, so it is useful evidence even when NER missed the whole span.
    if has_name_shape:
        return "PER", "name_shape"

    # The nearest legal-structure cue is more reliable than counting every
    # cue in a broad window. For example, a defendant may be immediately
    # followed by an address containing several location cues.
    if person_distance is not None and person_distance <= max_cue_distance and (
        location_distance is None
        or location_distance > max_cue_distance
        or person_distance <= location_distance
    ):
        return "PER", "nearest_person_context"
    if location_distance is not None and location_distance <= max_cue_distance:
        return "LOC", "location_context"
    if label in {"PER", "LOC"}:
        return label, "ner_label"
    if label == "ORG":
        return label, "ner_label"
    return None, "unresolved_context"


def word_spans(text, start, end):
    return list(WORD_RE.finditer(text[max(0, start):min(len(text), end)]))


def expand_marker_span(text, marker_match):
    """Expand ``Nguyễn Thị L1`` while leaving ``xã M`` as just ``M``."""
    start, end = marker_match.span(1)
    window_start = max(0, start - 100)
    words = list(WORD_RE.finditer(text[window_start:end]))
    if not words:
        return start, end

    marker_word_index = len(words) - 1
    for index, word in enumerate(words):
        if window_start + word.start(0) <= start < window_start + word.end(0):
            marker_word_index = index
            break

    selected_start = start
    included = 0
    for index in range(marker_word_index - 1, -1, -1):
        previous = words[index]
        current = words[index + 1]
        previous_end = window_start + previous.end(0)
        current_start = window_start + current.start(0)
        gap = text[previous_end:current_start]
        token = previous.group(0)
        if gap.strip() or not token or not token[0].isupper() or not is_name_prefix_token(token):
            break
        # Avoid swallowing headings or a preceding sentence's proper noun.
        if token.casefold() in {"page", "trang"}:
            break
        selected_start = window_start + previous.start(0)
        included += 1
        if included >= 3:
            break
    return selected_start, end


def trim_replacement_span(text, start, end, label):
    """Keep titles and administrative units outside the replacement span."""
    value = text[start:end]
    if label == "PER":
        title = TITLE_RE.match(value)
        if title:
            start += title.end()
    elif label == "LOC":
        prefix = LOCATION_PREFIX_RE.match(value)
        if prefix:
            start += prefix.end()
    elif label in {"ORG", "ADDR"}:
        # Replace only the masked value.  Keep ``Ngân hàng ...`` and ``Số``
        # visible so the generated text retains the document's structure.
        matches = list(MARKER_RE.finditer(value))
        if matches:
            marker_match = matches[-1]
            start += marker_match.start(1)
            end = start + (marker_match.end(1) - marker_match.start(1))
    return start, end


def is_spurious_marker(text, marker_start, marker_end):
    """Reject obvious abbreviations and administrative OCR artifacts.

    A one-letter marker is useful only when it is a real anonymization token.
    In particular, ``Thành phốM`` is usually an attached administrative word,
    while ``H. Định Quán`` uses ``H.`` as an abbreviation. Do not reject
    ``tỉnh H.`` or ``Nguyễn Văn H.`` because those can be valid masked forms.
    """
    before = text[max(0, marker_start - 40):marker_start]
    compact = marker_start > 0 and text[marker_start - 1].isalpha()
    if compact and ADMINISTRATIVE_PREFIX_RE.search(before):
        return True

    after = text[marker_end:marker_end + 1]
    if after == "." and not (
        ADMINISTRATIVE_PREFIX_RE.search(before)
        or PERSON_MARKER_PREFIX_RE.search(before)
        or (
            preceding_cue_distance(text, marker_start, ORGANIZATION_CUE_RE) is not None
            and preceding_cue_distance(text, marker_start, ORGANIZATION_CUE_RE) <= 60
        )
    ):
        return True
    return False


def marker_span_in_span(text, start, end):
    matches = [
        match for match in MARKER_RE.finditer(text[start:end])
        if not is_spurious_marker(
            text,
            start + match.start(1),
            start + match.end(1),
        )
    ]
    if not matches:
        matches = [
            match for match in COMPACT_MARKER_RE.finditer(text[start:end])
            if match.start(1) > 0 and text[start + match.start(1) - 1].islower()
            and not is_spurious_marker(
                text,
                start + match.start(1),
                start + match.end(1),
            )
        ]
    if not matches:
        return None
    match = matches[-1]
    return (
        start + match.start(1),
        start + match.end(1),
        marker_value(match.group(1)),
    )


def marker_in_span(text, start, end):
    span = marker_span_in_span(text, start, end)
    return span[2] if span else None


def overlapping(mention, start, end, label):
    return mention["label"] == label and mention["start"] < end and mention["end"] > start


def add_or_merge_mention(mentions, candidate):
    for existing in mentions:
        if (
            existing["label"] == candidate["label"]
            and existing["start"] == candidate["start"]
            and existing["end"] == candidate["end"]
        ):
            existing["detectors"] = sorted(set(existing["detectors"]) | set(candidate["detectors"]))
            if existing.get("score") is None and candidate.get("score") is not None:
                existing["score"] = candidate["score"]
            if existing.get("marker") is None:
                existing["marker"] = candidate.get("marker")
            if existing.get("link_evidence") == "marker_regex" and candidate.get("link_evidence"):
                existing["link_evidence"] = candidate["link_evidence"]
            return existing
    mentions.append(candidate)
    return candidate


def collect_mentions(text, ner_record):
    """Union NER predictions and full-text marker candidates."""
    mentions = []
    ner_entities = (ner_record or {}).get("entities", [])

    for entity in ner_entities:
        model_label = canonical_label(entity.get("label"))
        if model_label not in {"PER", "ORG", "LOC"}:
            continue
        start = max(0, min(len(text), int(entity.get("start", 0))))
        end = max(start, min(len(text), int(entity.get("end", start))))
        if end <= start:
            continue
        marker = marker_in_span(text, start, end)
        marker_span = marker_span_in_span(text, start, end) if marker else None
        # Rules own structured address spans.  NER is the primary detector for
        # people and organizations; LOC NER is retained only as a fallback for
        # free-form locations outside those structured spans.
        address_role = structured_address_role(text, start, end)
        if marker_span:
            # For a compound NER span such as ``Số 10/11 đường N``, classify
            # the masked value by the cue immediately before the marker, not
            # by the first cue in the whole span.
            address_role = structured_address_role(text, marker_span[0], marker_span[1])
        if address_role in {"house_number", "floor", "room", "parcel"}:
            label = "ADDR"
            link_evidence = "address_rule_overrides_ner" if marker else "address_rule"
        elif address_role:
            label = "LOC"
            link_evidence = f"address_rule_overrides_ner:{address_role}" if marker else f"address_rule:{address_role}"
        else:
            label = model_label
            if model_label == "LOC":
                link_evidence = "ner_loc_fallback"
            else:
                link_evidence = "ner_per_org" if model_label in {"PER", "ORG"} else "ner"
        if marker and link_evidence == "ner_per_org":
            link_evidence = "ner_marker_per_org"
        mention_start, mention_end = start, end
        if marker_span and (
            label in {"ORG", "ADDR"}
            or (label == "LOC" and address_role is not None)
        ):
            # Preserve visible cues (``Ngân hàng``, ``Số``, ``đường``, ...)
            # and replace only the masked value.
            mention_start, mention_end = marker_span[0], marker_span[1]
        mention = {
            "text": text[mention_start:mention_end],
            "start": mention_start,
            "end": mention_end,
            "label": label,
            "marker": marker,
            "score": entity.get("score"),
            "detectors": ["ner"],
            "link_evidence": link_evidence,
        }
        add_or_merge_mention(mentions, mention)

    # Scan both forms, but de-duplicate compact matches that are also part of
    # a normal spaced match.
    marker_matches = [
        match for match in MARKER_RE.finditer(text)
        if not is_spurious_marker(text, match.start(1), match.end(1))
    ]
    marker_matches.extend(
        match for match in COMPACT_MARKER_RE.finditer(text)
        if match.start(1) > 0
        and text[match.start(1) - 1].islower()
        and not is_spurious_marker(text, match.start(1), match.end(1))
        and not any(match.start(1) == normal.start(1) for normal in marker_matches)
    )
    known_marker_labels = defaultdict(set)
    for mention in mentions:
        marker = marker_value(mention.get("marker"))
        if marker:
            known_marker_labels[marker].add(mention["label"])

    candidate_records = []
    for match in marker_matches:
        marker = marker_value(match.group(1))
        if not marker:
            continue
        raw_start, raw_end = expand_marker_span(text, match)
        overlapping_mentions = [
            mention for mention in mentions
            if mention["start"] <= raw_end and mention["end"] >= raw_start
            and marker_in_span(text, mention["start"], mention["end"]) == marker
        ]
        ner_label = overlapping_mentions[0]["label"] if overlapping_mentions else None
        # Classify using the complete recovered mention span. For
        # ``bị cáo Nguyễn Văn H, ... xã M`` the person cue is near the start
        # of the mention, while the address cue belongs to the next clause.
        label, evidence = classify_marker(text, raw_start, raw_end, ner_label)
        candidate_records.append((match, marker, raw_start, raw_end, overlapping_mentions, label, evidence))
        if label in {"PER", "ORG", "LOC", "ADDR"}:
            known_marker_labels[marker].add(label)

    # A bare occurrence may appear far from its role cue. Once the document
    # has established that marker H is a person marker, later bare H mentions
    # can be recovered without requiring another nearby "ông/bị cáo" cue.
    for match, marker, raw_start, raw_end, overlapping_mentions, label, evidence in candidate_records:
        if label not in {"PER", "ORG", "LOC", "ADDR"} and len(known_marker_labels[marker]) == 1:
            label = next(iter(known_marker_labels[marker]))
            evidence = "known_marker_propagation"
        if label not in {"PER", "ORG", "LOC", "ADDR"}:
            continue
        if overlapping_mentions:
            # Retain the NER span because it usually contains the complete
            # anonymized name, not merely its final marker.
            selected = max(overlapping_mentions, key=lambda item: item["end"] - item["start"])
            if label in {"ADDR", "LOC"} and label != selected["label"]:
                selected["label"] = label
            selected["detectors"] = sorted(set(selected["detectors"]) | {"marker_regex"})
            selected["marker"] = marker
            selected["link_evidence"] = "ner_and_marker_regex"
            continue

        start, end = trim_replacement_span(text, raw_start, raw_end, label)
        add_or_merge_mention(mentions, {
            "text": text[start:end],
            "start": start,
            "end": end,
            "label": label,
            "marker": marker,
            "score": None,
            "detectors": ["marker_regex"],
            "link_evidence": evidence,
        })

    return sorted(mentions, key=lambda item: (item["start"], item["end"], item["label"]))


def infer_gender(text, mentions):
    female = 0
    male = 0
    for mention in mentions:
        start, end = mention["start"], mention["end"]
        surface = text[start:end]
        # Middle names are strong local evidence when the public document
        # retains them: Nguyễn Thị L is much more likely female, while
        # Nguyễn Văn L is much more likely male.
        has_female_middle = bool(re.search(r"\bThị\b", surface))
        has_male_middle = bool(re.search(r"\bVăn\b", surface))
        if has_female_middle and not has_male_middle:
            female += 2
            continue
        if has_male_middle and not has_female_middle:
            male += 2
            continue

        left = max(0, start - 45)
        right = min(len(text), end + 45)
        local = text[left:right]
        female_distances = []
        male_distances = []
        for match in FEMALE_GENDER_CUE_RE.finditer(local):
            cue_start = left + match.start()
            cue_end = left + match.end()
            female_distances.append(0 if cue_start <= end and cue_end >= start else min(abs(start - cue_end), abs(cue_start - end)))
        for match in MALE_GENDER_CUE_RE.finditer(local):
            cue_start = left + match.start()
            cue_end = left + match.end()
            male_distances.append(0 if cue_start <= end and cue_end >= start else min(abs(start - cue_end), abs(cue_start - end)))
        nearest_female = min(female_distances) if female_distances else None
        nearest_male = min(male_distances) if male_distances else None
        if nearest_female is not None and (nearest_male is None or nearest_female < nearest_male):
            female += 1
        elif nearest_male is not None:
            male += 1
    if female > male:
        return "female"
    if male > female:
        return "male"
    return "male"


def stable_slot(doc_id, marker):
    digest = hashlib.sha256(f"{doc_id}|{marker}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def choose_given_name(marker, gender, used, doc_id):
    initial = marker_initial(marker)
    primary_pool = GIVEN_NAMES[gender].get(initial, [])
    fallback_pool = FALLBACK_GIVEN_NAMES[gender]
    # J, W, Z, F, and similar codes can be arbitrary publication markers,
    # rather than Vietnamese given-name initials. Use a common Vietnamese
    # fallback instead of inventing strings such as ``J Minh``.
    preferred = marker_index(marker)
    slot = preferred + stable_slot(doc_id, marker)

    # Prefer a common name beginning with the marker. Only use the fallback
    # pool after all suitable marker-initial names are already used.
    pools = [primary_pool] if primary_pool else []
    pools.append([name for name in fallback_pool if name not in primary_pool])
    for pool in pools:
        if not pool:
            continue
        start = slot % len(pool)
        for offset in range(len(pool)):
            value = pool[(start + offset) % len(pool)]
            if value not in used:
                used.add(value)
                return value
    # Never append a number: that creates an unnatural given name. Reusing a
    # common name is preferable for synthetic data once the pool is exhausted.
    value = fallback_pool[(preferred + stable_slot(doc_id, marker)) % len(fallback_pool)]
    used.add(value)
    return value


def marker_initial_preserved(marker, synthetic_value):
    if not marker or not synthetic_value:
        return None
    given_name = str(synthetic_value).split()[-1]
    return bool(given_name) and initial_key(given_name) == initial_key(marker)


def person_prefix_tokens(surface):
    marker_matches = list(MARKER_RE.finditer(surface))
    if not marker_matches:
        marker_matches = list(COMPACT_MARKER_RE.finditer(surface))
    if not marker_matches:
        return []

    prefix = surface[:marker_matches[-1].start(1)].strip()
    prefix = TITLE_RE.sub("", prefix).strip()
    tokens = prefix.split()

    # OCR occasionally splits a surname, for example ``Nguy ễn``. Rejoin
    # adjacent tokens only when the result is a known Vietnamese family name.
    family_name_keys = {normalize_for_match(name): name for name in FAMILY_NAMES}
    index = 0
    normalized_tokens = []
    while index < len(tokens):
        if index + 1 < len(tokens):
            joined = normalize_for_match(tokens[index] + tokens[index + 1])
            if joined in family_name_keys:
                normalized_tokens.append(family_name_keys[joined])
                index += 2
                continue
        normalized_tokens.append(tokens[index])
        index += 1
    return normalized_tokens


def person_mention_structure_score(mention, text):
    surface = text[mention["start"]:mention["end"]]
    tokens = person_prefix_tokens(surface)
    family_name_keys = {normalize_for_match(name) for name in FAMILY_NAMES}
    has_family = any(normalize_for_match(token) in family_name_keys for token in tokens)
    has_middle = any(normalize_for_match(token) in {"thị", "văn"} for token in tokens)
    # If multiple mentions have the same structure, keep the first occurrence
    # in the document. This avoids allowing a later OCR-expanded span to
    # overwrite the document's original visible surname/middle-name pattern.
    return (int(has_family), int(has_middle), len(tokens), -mention["start"])


def synthetic_person_name(doc_id, marker, mentions, text, used):
    # Prefer a structurally valid visible name prefix over a merely longer
    # OCR span. This preserves forms such as ``Nguy ễn Văn M`` as Nguyễn Văn M.
    best = max(mentions, key=lambda item: person_mention_structure_score(item, text))
    surface = text[best["start"]:best["end"]]
    prefix_tokens = person_prefix_tokens(surface)
    prefix = " ".join(prefix_tokens)
    if re.search(r"\bThị\b", prefix):
        gender = "female"
    elif re.search(r"\bVăn\b", prefix):
        gender = "male"
    else:
        gender = infer_gender(text, mentions)
    given = choose_given_name(marker, gender, used, doc_id)
    family_name_keys = {normalize_for_match(name) for name in FAMILY_NAMES}
    has_family_prefix = any(normalize_for_match(token) in family_name_keys for token in prefix_tokens)
    if prefix_tokens and has_family_prefix and all(token[:1].isupper() for token in prefix_tokens):
        if normalize_for_match(prefix_tokens[-1]) == normalize_for_match(given):
            given = choose_given_name(marker, gender, used, doc_id)
        return " ".join(prefix_tokens + [given])

    family = FAMILY_NAMES[(stable_slot(doc_id, marker) + marker_index(marker)) % len(FAMILY_NAMES)]
    middle = "Thị" if gender == "female" else "Văn"
    return f"{family} {middle} {given}"


def location_unit(text, mentions):
    joined = " ".join(text[max(0, m["start"] - 40):m["end"]] for m in mentions)
    if re.search(r"\b(?:tỉnh|thành\s+phố)\b", joined, re.IGNORECASE):
        return "province"
    if re.search(r"\b(?:huyện|quận|thị\s+xã)\b", joined, re.IGNORECASE):
        return "district"
    if re.search(r"\b(?:xã|phường|thị\s+trấn)\b", joined, re.IGNORECASE):
        return "commune"
    if re.search(r"\b(?:ấp|thôn|khu\s+phố|tổ\s+dân\s+phố|khu\s+dân\s+cư)\b", joined, re.IGNORECASE):
        return "hamlet"
    return "generic"


def synthetic_location(doc_id, marker, mentions, text, used):
    role = mention_role(text, mentions[0]) if mentions else "location:freeform"
    unit = role.split(":", 1)[1] if role.startswith("location:") else location_unit(text, mentions)
    if unit not in LOCATION_NAMES:
        unit = location_unit(text, mentions)
    initial = marker_initial(marker)
    pool = LOCATION_NAMES[unit].get(initial) or LOCATION_NAMES["generic"].get(initial)
    if not pool:
        # Publication markers are not guaranteed to be place-name initials.
        # Fall back to a real-looking name from the same administrative unit;
        # never emit artificial values such as ``K An`` or ``X An``.
        pool = list(dict.fromkeys(
            value
            for values in LOCATION_NAMES[unit].values()
            for value in values
        ))
    if not pool:
        pool = list(dict.fromkeys(
            value
            for values in LOCATION_NAMES["generic"].values()
            for value in values
        ))
    start = (marker_index(marker) + stable_slot(doc_id, marker)) % len(pool)
    for offset in range(len(pool)):
        value = pool[(start + offset) % len(pool)]
        if value not in used:
            used.add(value)
            return value
    # Reuse a plausible location rather than adding an unnatural numeric
    # suffix after the small synthetic pool is exhausted.
    return pool[start]


def synthetic_organization(doc_id, marker, used):
    """Create a plausible organization suffix for a masked organization."""
    start = (marker_index(marker) + stable_slot(doc_id, f"ORG|{marker}")) % len(SYNTHETIC_ORGANIZATION_NAMES)
    for offset in range(len(SYNTHETIC_ORGANIZATION_NAMES)):
        value = SYNTHETIC_ORGANIZATION_NAMES[(start + offset) % len(SYNTHETIC_ORGANIZATION_NAMES)]
        if value not in used:
            used.add(value)
            return value
    return SYNTHETIC_ORGANIZATION_NAMES[start]


def synthetic_address_component(doc_id, marker, mentions, text, used):
    """Create a plausible address number without pretending to recover it."""
    role = mention_role(text, mentions[0]) if mentions else "house_number"
    # Keep each structured slot plausible: a floor is not a three-digit house
    # number, and a room/parcel can use a wider range.  The marker is a
    # publication placeholder, not evidence of the original numeric value.
    if role == "floor":
        lower, size = 1, 30
    elif role in {"room", "parcel"}:
        lower, size = 1, 999
    else:
        lower, size = 10, 890
    value = str(lower + (stable_slot(doc_id, f"ADDR|{role}|{marker}") % size))
    if value not in used:
        used.add(value)
        return value
    for offset in range(1, size):
        candidate = str(lower + ((int(value) - lower + offset) % size))
        if candidate not in used:
            used.add(candidate)
            return candidate
    return value


def link_document(doc_id, text, ner_record):
    mentions = collect_mentions(text, ner_record)
    groups = defaultdict(list)
    for mention in mentions:
        role = mention_role(text, mention)
        mention["role"] = role
        marker = marker_value(mention.get("marker"))
        if marker:
            # A marker is not a globally unique entity.  ``X`` in a house
            # number, hamlet, and organization name must be linked separately.
            key = (mention["label"], marker, role)
        else:
            # Exact unmarked mentions are linked for audit purposes, but are
            # not replaced because they are not demonstrably anonymized.
            key = (
                mention["label"],
                "UNMARKED",
                role,
                normalize_for_match(mention["text"]),
            )
        groups[key].append(mention)

    ordered_groups = sorted(groups.values(), key=lambda group: min(m["start"] for m in group))
    used_names = set()
    used_locations = set()
    used_organizations = set()
    used_address_values = set()
    entities = []
    replacements = []

    for index, group in enumerate(ordered_groups, 1):
        first = min(group, key=lambda item: item["start"])
        marker = marker_value(first.get("marker"))
        entity_id = f"{first['label']}_{index:04d}"
        if marker and first["label"] == "PER":
            synthetic = synthetic_person_name(doc_id, marker, group, text, used_names)
        elif marker and first["label"] == "LOC":
            synthetic = synthetic_location(doc_id, marker, group, text, used_locations)
        elif marker and first["label"] == "ORG":
            synthetic = synthetic_organization(doc_id, marker, used_organizations)
        elif marker and first["label"] == "ADDR":
            synthetic = synthetic_address_component(doc_id, marker, group, text, used_address_values)
        else:
            synthetic = None

        if marker and first["label"] == "PER" and synthetic:
            name_rule = "preserve_marker_initial" if marker_initial_preserved(marker, synthetic) else "common_fallback"
        else:
            name_rule = None

        entity_mentions = []
        for mention in sorted(group, key=lambda item: (item["start"], item["end"])):
            mention["entity_id"] = entity_id
            entity_mentions.append({
                "text": mention["text"],
                "start": mention["start"],
                "end": mention["end"],
                "marker": mention.get("marker"),
                "role": mention.get("role"),
                "score": mention.get("score"),
                "detectors": mention["detectors"],
                "link_evidence": mention["link_evidence"],
            })
            if synthetic and marker:
                replacements.append({
                    "entity_id": entity_id,
                    "label": first["label"],
                    "start": mention["start"],
                    "end": mention["end"],
                    "replacement": synthetic,
                })

        entities.append({
            "entity_id": entity_id,
            "label": first["label"],
            "role": first.get("role"),
            "marker": marker,
            "synthetic_value": synthetic,
            "reconstructable": bool(synthetic and marker),
            "name_rule": name_rule,
            "marker_initial_preserved": marker_initial_preserved(marker, synthetic),
            "mentions": entity_mentions,
        })

    return mentions, entities, replacements


def apply_replacements(text, replacements):
    result = text
    applied = []
    original_length = len(text)
    occupied_start = len(text) + 1
    for replacement in sorted(replacements, key=lambda item: (item["start"], item["end"]), reverse=True):
        start, end = replacement["start"], replacement["end"]
        if end > occupied_start or start < 0 or end > original_length:
            continue
        result = result[:start] + replacement["replacement"] + result[end:]
        occupied_start = start
        applied.append(replacement)
    return result, list(reversed(applied))


def process_row(row_number, row, ner_record, text_field):
    doc_id = str(row.get("case_id", row.get("doc_name", row.get("id", row_number))))
    source_text = row.get(text_field)
    if source_text is None:
        return doc_id, dict(row), {"doc_id": doc_id, "error": f"missing field: {text_field}"}, None

    source_text = str(source_text)
    mentions, entities, replacements = link_document(doc_id, source_text, ner_record)
    synthetic_text, applied = apply_replacements(source_text, replacements)
    output_row = dict(row)
    output_row["synthetic_markdown"] = synthetic_text
    output_row["original_anonymized_markdown"] = source_text
    output_row["synthetic_reconstruction"] = True
    output_row["reconstruction_stats"] = {
        "mentions": len(mentions),
        "linked_entities": len(entities),
        "reconstructable_entities": sum(entity["reconstructable"] for entity in entities),
        "replacements": len(applied),
    }
    audit = {
        "row_number": row_number,
        "doc_id": doc_id,
        "entities": entities,
        "replacements": applied,
    }
    return doc_id, output_row, audit, {"mentions": mentions, "entities": entities, "replacements": applied}


def main():
    parser = argparse.ArgumentParser(description="Create synthetic un-anonymized legal data from NER and anonymization markers")
    parser.add_argument("--input-file", default=DEFAULT_SOURCE, help="Published anonymized source JSONL")
    parser.add_argument("--ner-file", default=DEFAULT_NER, help="NER prediction JSONL; may be incomplete")
    parser.add_argument("--output-file", default=DEFAULT_DATASET_OUTPUT, help="Synthetic dataset JSONL")
    parser.add_argument("--links-file", default=DEFAULT_LINK_OUTPUT, help="Entity-link audit JSONL")
    parser.add_argument("--maps-file", default=DEFAULT_MAP_OUTPUT, help="Replacement-map JSONL")
    parser.add_argument("--text-field", default="markdown", help="Source text field")
    parser.add_argument("--limit", type=int, default=0, help="Rows to process; 0 means all")
    parser.add_argument("--offset", type=int, default=0, help="Rows to skip")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        parser.error(f"Input file does not exist: {args.input_file}")
    if args.limit < 0 or args.offset < 0:
        parser.error("--limit and --offset cannot be negative")

    ner_by_id = load_jsonl_by_id(args.ner_file)
    for path in (args.output_file, args.links_file, args.maps_file):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    documents = 0
    errors = 0
    replacements = 0
    reconstructable = 0
    label_counts = Counter()

    with (
        open(args.output_file, "w", encoding="utf-8") as dataset_handle,
        open(args.links_file, "w", encoding="utf-8") as links_handle,
        open(args.maps_file, "w", encoding="utf-8") as maps_handle,
    ):
        for row_number, row in iter_source_rows(args.input_file, args.offset, args.limit):
            doc_id, output_row, audit, details = process_row(
                row_number, row, ner_by_id.get(str(row.get("case_id", row.get("doc_name", row.get("id", row_number))))), args.text_field
            )
            dataset_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            links_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            map_record = {
                "row_number": row_number,
                "doc_id": doc_id,
                "entities": [
                    {
                        "entity_id": entity["entity_id"],
                        "label": entity["label"],
                        "marker": entity["marker"],
                        "synthetic_value": entity["synthetic_value"],
                        "reconstructable": entity["reconstructable"],
                        "name_rule": entity["name_rule"],
                        "marker_initial_preserved": entity["marker_initial_preserved"],
                    }
                    for entity in audit.get("entities", [])
                ],
            }
            maps_handle.write(json.dumps(map_record, ensure_ascii=False) + "\n")

            documents += 1
            if "error" in audit:
                errors += 1
            else:
                replacements += len(audit["replacements"])
                reconstructable += sum(entity["reconstructable"] for entity in audit["entities"])
                label_counts.update(entity["label"] for entity in audit["entities"] if entity["reconstructable"])
            if documents == 1 or documents % 100 == 0:
                print(f"[RECON] Processed {documents} documents; replacements={replacements}")

    print("\n[RECON] Completed")
    print(f"[RECON] Documents: {documents}; errors: {errors}")
    print(f"[RECON] Reconstructable entities: {reconstructable}; by type: {dict(label_counts)}")
    print(f"[RECON] Replacements: {replacements}")
    print(f"[RECON] Dataset saved to: {args.output_file}")
    print(f"[RECON] Links saved to: {args.links_file}")
    print(f"[RECON] Maps saved to: {args.maps_file}")


if __name__ == "__main__":
    main()
