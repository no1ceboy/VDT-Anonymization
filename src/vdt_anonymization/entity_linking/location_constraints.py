"""High-confidence location disambiguation from explicit address context."""

from __future__ import annotations

import re

from .dataset import MENTION_CLOSE, MENTION_OPEN, normalize_text


_ABBREVIATION_PERIOD = "\ue000"
_ADMIN_ABBREVIATION = re.compile(r"\b(?:tp|p|q|h|tx)\.", re.IGNORECASE)
_CLAUSE_BOUNDARY = re.compile(r"[;.!?\n]+")
_PROVINCE = re.compile(
    r"(?=\b(?:tỉnh|thành\s+phố|tp\.?)\s+"
    r"([\wÀ-ỹĐđ]+(?:[-\s]+[\wÀ-ỹĐđ]+){0,3}))",
    re.IGNORECASE,
)
_PROVINCE_STOP_WORDS = {
    "và", "hoặc", "với", "thuộc", "tại", "tỉnh", "thành", "phố",
    "huyện", "quận", "xã", "phường", "thị", "trấn",
}


def explicit_province(mention: dict) -> str | None:
    """Return a province explicitly attached to this mention's address clause.

    Contexts can include several addresses. Only the punctuation-delimited
    clause containing the marked mention is considered; if it names multiple
    provinces, the result is intentionally unknown rather than guessed.
    """
    context = str(mention.get("context") or "")
    if MENTION_OPEN not in context or MENTION_CLOSE not in context:
        return None

    protected = _ADMIN_ABBREVIATION.sub(
        lambda match: match.group(0)[:-1] + _ABBREVIATION_PERIOD, context
    )
    clauses = _CLAUSE_BOUNDARY.split(protected)
    clause = next((part for part in clauses if MENTION_OPEN in part), None)
    if clause is None:
        return None

    plain = clause.replace(MENTION_OPEN, "").replace(MENTION_CLOSE, "")
    plain = plain.replace(_ABBREVIATION_PERIOD, ".")
    provinces = set()
    for match in _PROVINCE.finditer(plain):
        words = re.findall(r"[\wÀ-ỹĐđ]+", match.group(1))
        stop_at = next(
            (index for index, word in enumerate(words) if normalize_text(word) in _PROVINCE_STOP_WORDS),
            len(words),
        )
        province = normalize_text(" ".join(words[:stop_at]))
        if province:
            provinces.add(province)
    provinces.discard("")
    return next(iter(provinces)) if len(provinces) == 1 else None


def has_explicit_province_conflict(pair: dict) -> bool:
    """Whether two location mentions have explicit, different provinces."""
    a = pair.get("mention_a") or {}
    b = pair.get("mention_b") or {}
    if str(a.get("label") or "").upper() != "LOC" or str(b.get("label") or "").upper() != "LOC":
        return False
    province_a, province_b = explicit_province(a), explicit_province(b)
    return province_a is not None and province_b is not None and province_a != province_b


def location_province_features(a: dict, b: dict) -> dict[str, float]:
    """Features for learning location hierarchy agreement/conflict."""
    if str(a.get("label") or "").upper() != "LOC" or str(b.get("label") or "").upper() != "LOC":
        return {"same_explicit_province": 0.0, "conflicting_explicit_province": 0.0}
    province_a, province_b = explicit_province(a), explicit_province(b)
    if province_a is None or province_b is None:
        return {"same_explicit_province": 0.0, "conflicting_explicit_province": 0.0}
    return {
        "same_explicit_province": float(province_a == province_b),
        "conflicting_explicit_province": float(province_a != province_b),
    }
