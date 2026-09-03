"""Build a synthetic un-anonymized legal dataset from NER predictions.

This is not identity recovery. It creates plausible, deterministic fake
identities for anonymized mentions in already published legal documents.

The pipeline is deliberately recall-oriented:

1. Read normal PER/LOC predictions from ``run_ner.py``.
2. Scan the complete source text for court-style markers such as ``M``, ``L``
   and ``L1`` that NER may miss.
3. Classify marker candidates using nearby person/location structure.
4. Link identical markers within a document and entity type.
5. Generate one synthetic name/location per linked entity.
6. Replace linked mentions from right to left.

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
    r"(?:ông|bà|anh|chị|em|cô|chú|bác|con|vợ|chồng|cha|mẹ|"
    r"bị\s+cáo|nguyên\s+đơn|bị\s+đơn|người\s+bị\s+hại|"
    r"người\s+có\s+quyền\s+lợi|người\s+liên\s+quan|người\s+làm\s+chứng|"
    r"luật\s+sư|đại\s+diện|đương\s+sự|bị\s+can|người\s+khởi\s+kiện)",
    re.IGNORECASE,
)
LOCATION_CUE_RE = re.compile(
    r"(?:xã|phường|thị\s+trấn|huyện|quận|thị\s+xã|thành\s+phố|tỉnh|"
    r"ấp|thôn|khu\s+phố|đường|ngõ|hẻm|khu\s+dân\s+cư|tòa\s+án|tand)",
    re.IGNORECASE,
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

ENTITY_ALIASES = {
    "PER": "PER",
    "PERSON": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
}

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

FAMILY_NAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Võ", "Vũ", "Đặng", "Bùi"]

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


def classify_marker(text, start, end, ner_label=None):
    label = canonical_label(ner_label)

    person_distance = nearest_cue_distance(text, start, end, PERSON_CUE_RE)
    location_distance = nearest_cue_distance(text, start, end, LOCATION_CUE_RE)
    max_cue_distance = 24
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
    return start, end


def marker_in_span(text, start, end):
    matches = list(MARKER_RE.finditer(text[start:end]))
    if not matches:
        matches = [
            match for match in COMPACT_MARKER_RE.finditer(text[start:end])
            if match.start(1) > 0 and text[start + match.start(1) - 1].islower()
        ]
    if not matches:
        return None
    match = matches[-1]
    return marker_value(match.group(1))


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
        label = canonical_label(entity.get("label"))
        if label not in {"PER", "LOC"}:
            continue
        start = max(0, min(len(text), int(entity.get("start", 0))))
        end = max(start, min(len(text), int(entity.get("end", start))))
        if end <= start:
            continue
        marker = marker_in_span(text, start, end)
        link_evidence = "ner_marker" if marker else "ner"
        mention = {
            "text": text[start:end],
            "start": start,
            "end": end,
            "label": label,
            "marker": marker,
            "score": entity.get("score"),
            "detectors": ["ner"],
            "link_evidence": link_evidence,
        }
        add_or_merge_mention(mentions, mention)

    # Scan both forms, but de-duplicate compact matches that are also part of
    # a normal spaced match.
    marker_matches = list(MARKER_RE.finditer(text))
    marker_matches.extend(
        match for match in COMPACT_MARKER_RE.finditer(text)
        if match.start(1) > 0
        and text[match.start(1) - 1].islower()
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
        if label in {"PER", "LOC"}:
            known_marker_labels[marker].add(label)

    # A bare occurrence may appear far from its role cue. Once the document
    # has established that marker H is a person marker, later bare H mentions
    # can be recovered without requiring another nearby "ông/bị cáo" cue.
    for match, marker, raw_start, raw_end, overlapping_mentions, label, evidence in candidate_records:
        if label not in {"PER", "LOC"} and len(known_marker_labels[marker]) == 1:
            label = next(iter(known_marker_labels[marker]))
            evidence = "known_marker_propagation"
        if label not in {"PER", "LOC"}:
            continue
        if overlapping_mentions:
            # Retain the NER span because it usually contains the complete
            # anonymized name, not merely its final marker.
            selected = max(overlapping_mentions, key=lambda item: item["end"] - item["start"])
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
    pool = primary_pool + [name for name in fallback_pool if name not in primary_pool]
    preferred = marker_index(marker)
    start = (preferred + stable_slot(doc_id, marker)) % len(pool)
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
    return bool(given_name) and given_name[0].upper() == marker_initial(marker)


def synthetic_person_name(doc_id, marker, mentions, text, used):
    # Preserve visible family/middle-name structure from the longest mention.
    best = max(mentions, key=lambda item: item["end"] - item["start"])
    surface = text[best["start"]:best["end"]]
    marker_match = list(MARKER_RE.finditer(surface)) or list(COMPACT_MARKER_RE.finditer(surface))
    prefix = ""
    if marker_match:
        prefix = surface[:marker_match[-1].start(1)].strip()
        prefix = TITLE_RE.sub("", prefix).strip()
    prefix_tokens = prefix.split()
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
    return "generic"


def synthetic_location(doc_id, marker, mentions, text, used):
    unit = location_unit(text, mentions)
    initial = marker_initial(marker)
    pool = LOCATION_NAMES[unit].get(initial) or LOCATION_NAMES["generic"].get(initial)
    if not pool:
        pool = [f"{initial} An"]
    start = (marker_index(marker) + stable_slot(doc_id, marker)) % len(pool)
    value = pool[start]
    if value in used:
        value = f"{value} {marker_index(marker) + 1}"
    used.add(value)
    return value


def link_document(doc_id, text, ner_record):
    mentions = collect_mentions(text, ner_record)
    groups = defaultdict(list)
    for mention in mentions:
        marker = marker_value(mention.get("marker"))
        if marker:
            key = (mention["label"], marker)
        else:
            # Exact unmarked mentions are linked for audit purposes, but are
            # not replaced because they are not demonstrably anonymized.
            key = (mention["label"], "UNMARKED", normalize_for_match(mention["text"]))
        groups[key].append(mention)

    ordered_groups = sorted(groups.values(), key=lambda group: min(m["start"] for m in group))
    used_names = set()
    used_locations = set()
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
