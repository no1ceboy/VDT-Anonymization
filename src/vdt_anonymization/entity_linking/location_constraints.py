"""High-confidence location disambiguation from explicit address context."""

from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .dataset import MENTION_CLOSE, MENTION_OPEN


_UNITS_PATH = Path(__file__).resolve().parents[1] / "resources" / "dvhcvn" / "units.tsv"
_CLAUSE_BOUNDARY = re.compile(r"[;.!?\r\n]+")
_TP_PERIOD = re.compile(r"\btp\.", re.IGNORECASE)
_PROVINCE_PREFIX = re.compile(r"(?<!\w)(?:tinh|thanh\s+pho|tp)(?!\w)\s*[:,\-]?\s*", re.IGNORECASE)
_DASHES = str.maketrans({"-": " ", "–": " ", "—": " ", "−": " "})


def _fold(value: object) -> str:
    """Normalize accents, case, dash variants, and spacing for lookup."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold().translate(_DASHES))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(text.split())


@lru_cache(maxsize=1)
def _province_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    """Compile strict aliases from the bundled, pinned province vocabulary."""
    with _UNITS_PATH.open(encoding="utf-8", newline="") as handle:
        provinces = [row["name"] for row in csv.DictReader(handle, delimiter="\t")
                     if row.get("level") == "1" and row.get("name")]

    aliases: dict[str, set[str]] = {}
    for name in provinces:
        canonical = " ".join(name.casefold().translate(_DASHES).split())
        aliases.setdefault(_fold(name), set()).add(canonical)

    # A common official-document abbreviation. Keep it scoped to an explicit
    # TP prefix; bare "HCM" is not enough evidence to infer a province.
    aliases.setdefault("hcm", set()).add("hồ chí minh")

    patterns = []
    for alias, canonical_names in aliases.items():
        # Never use an accent-insensitive spelling if it could identify more
        # than one province in the official snapshot.
        if len(canonical_names) != 1:
            continue
        words = alias.split()
        body = r"\s+".join(re.escape(word) for word in words)
        patterns.append((re.compile(body + r"(?!\w)", re.IGNORECASE), next(iter(canonical_names))))
    return tuple(sorted(patterns, key=lambda item: len(item[0].pattern), reverse=True))


def explicit_province(mention: dict) -> str | None:
    """Return a province only when explicitly named in the mention's clause.

    The bundled vocabulary bounds the match at the official province name, so
    nearby OCR or prose (for example ``Vĩnh Phúc có hiệu lực``) cannot become
    part of the extracted value. If the marked clause contains an unknown or
    conflicting province reference, return ``None`` rather than guessing.
    """
    context = str(mention.get("context") or "")
    if MENTION_OPEN not in context or MENTION_CLOSE not in context:
        return None

    # Keep the period in "TP." from being mistaken for the end of a clause.
    protected = _TP_PERIOD.sub("tp ", context)
    clause = next((part for part in _CLAUSE_BOUNDARY.split(protected)
                   if MENTION_OPEN in part and MENTION_CLOSE in part), None)
    if clause is None:
        return None

    plain = clause.replace(MENTION_OPEN, "").replace(MENTION_CLOSE, "")
    normalized = _fold(plain)
    provinces: set[str] = set()
    prefix_count = 0
    for prefix in _PROVINCE_PREFIX.finditer(normalized):
        prefix_count += 1
        tail = normalized[prefix.end():]
        matched = next(((match, canonical) for pattern, canonical in _province_patterns()
                        if (match := pattern.match(tail))), None)
        if matched is None:
            return None
        _, canonical = matched
        provinces.add(canonical)

    if prefix_count == 0 or len(provinces) != 1:
        return None
    return next(iter(provinces))


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
