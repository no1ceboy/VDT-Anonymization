"""Rule-first Vietnamese location resolution for document entity linking.

The NER label remains authoritative. For LOC mentions, this module uses the
bundled administrative gazetteer and Vietnamese address-unit words to recover
the most specific location and its parent path. Exact canonical matches can
link; explicit hierarchy conflicts block; uncertain parses are left to the
learned pair scorer.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

from ..core.gazetteer import ancestor_chain, records as gazetteer_records
from .dataset import MENTION_CLOSE, MENTION_OPEN


_CLAUSE_BOUNDARY = re.compile(r"[;.!?]+|(?:\r?\n[ \t]*){2,}")
_TP_PERIOD = re.compile(r"(?<!\w)tp\.", re.IGNORECASE)
_DASHES = str.maketrans({"-": " ", "\u2010": " ", "\u2011": " ", "\u2012": " ", "\u2013": " ", "\u2014": " "})
_OPEN_SENTINEL = "codexmentionopenmarker"
_CLOSE_SENTINEL = "codexmentionclosemarker"

# These are grammatical address cues, not proof that a name is a location.
# The gazetteer resolves province/district/commune names; lower units are
# represented by their type and local alias, constrained by known parents.
_PREFIX_LEVELS = {
    "tinh": ("province",),
    "thanh pho": ("province", "district"),
    "tp": ("province", "district"),
    "huyen": ("district",),
    "quan": ("district",),
    "thi xa": ("district",),
    "thi tran": ("commune",),
    "xa": ("commune",),
    "phuong": ("commune",),
    "ap": ("hamlet",),
    "thon": ("hamlet",),
    "xom": ("hamlet",),
    "ban": ("hamlet",),
    "buon": ("hamlet",),
    "bon": ("hamlet",),
    "lang": ("hamlet",),
    "khom": ("hamlet",),
    "phum": ("hamlet",),
    "soc": ("hamlet",),
    "khu pho": ("hamlet",),
    "khu dan cu": ("hamlet",),
    "cum dan cu": ("hamlet",),
    "khu": ("hamlet",),
    "to dan pho": ("hamlet",),
    "to": ("hamlet",),
}
_PREFIX_UNITS = {
    "tinh": {"tỉnh"},
    "thanh pho": {"thành phố"},
    "tp": {"thành phố"},
    "huyen": {"huyện"},
    "quan": {"quận"},
    "thi xa": {"thị xã"},
    "thi tran": {"thị trấn"},
    "xa": {"xã"},
    "phuong": {"phường"},
}
_PREFIX_RE = re.compile(
    r"(?<!\w)(?P<prefix>to\s+dan\s+pho|cum\s+dan\s+cu|khu\s+dan\s+cu|thanh\s+pho|"
    r"thi\s+tran|thi\s+xa|khu\s+pho|tinh|huyen|quan|xa|phuong|tp|ap|thon|xom|ban|"
    r"buon|bon|lang|khom|phum|soc|khu|to)(?!\w)\s*[:,\-]?\s*",
    re.IGNORECASE,
)
_OCR_UNIT_REPAIRS = (
    (re.compile(r"\bthanh\s+ph\s+o\b"), "thanh pho"),
    (re.compile(r"\bhu\s+yen\b"), "huyen"),
    (re.compile(r"\bhuy\s+en\b"), "huyen"),
    (re.compile(r"\bth\s+i\s+xa\b"), "thi xa"),
    (re.compile(r"\bthi\s+x\s+a\b"), "thi xa"),
    (re.compile(r"\bth\s+i\s+tr\s+an\b"), "thi tran"),
    (re.compile(r"\bqu\s+an\b"), "quan"),
    (re.compile(r"\bph\s+uong\b"), "phuong"),
    (re.compile(r"\bt\s+inh\b"), "tinh"),
    (re.compile(r"\bti\s+nh\b"), "tinh"),
    (re.compile(r"\bx\s+a\b"), "xa"),
)
_COMPONENT_BOUNDARY = re.compile(r"[,;.!?\r\n]+|\b(?:va|hoac|hay)\b")
_ROLE_BY_LEVEL = {"1": "province", "2": "district", "3": "commune"}
_LEVEL_BY_ROLE = {value: key for key, value in _ROLE_BY_LEVEL.items()}


def _fold(value: object) -> str:
    """Accent-, case-, dash-, and whitespace-normalized lookup form."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold().translate(_DASHES))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    for pattern, replacement in _OCR_UNIT_REPAIRS:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


def _without_markers(value: str) -> str:
    return (value.replace(MENTION_OPEN.casefold(), "").replace(MENTION_CLOSE.casefold(), "")
            .replace(_OPEN_SENTINEL, "").replace(_CLOSE_SENTINEL, ""))


def _marked_clause(mention: dict) -> tuple[str, tuple[int, int], str] | None:
    """Return folded clause, target span in it, and folded target surface."""
    context = str(mention.get("context") or "")
    open_marker, close_marker = MENTION_OPEN.casefold(), MENTION_CLOSE.casefold()
    if open_marker not in context.casefold() or close_marker not in context.casefold():
        return None
    protected = _TP_PERIOD.sub("tp ", context)
    for raw_clause in _CLAUSE_BOUNDARY.split(protected):
        if open_marker not in raw_clause.casefold() or close_marker not in raw_clause.casefold():
            continue
        tagged = raw_clause.replace(MENTION_OPEN, _OPEN_SENTINEL).replace(MENTION_CLOSE, _CLOSE_SENTINEL)
        folded = _fold(tagged)
        start = folded.find(_OPEN_SENTINEL)
        end = folded.find(_CLOSE_SENTINEL, start + len(_OPEN_SENTINEL)) if start >= 0 else -1
        if start < 0 or end < 0:
            continue
        target_start = start + len(_OPEN_SENTINEL)
        target = folded[target_start:end]
        return folded, (target_start, end), _without_markers(target).strip()
    return None


@lru_cache(maxsize=1)
def _rows_by_role() -> dict[str, tuple[dict, ...]]:
    rows: dict[str, list[dict]] = {role: [] for role in _LEVEL_BY_ROLE}
    for row in gazetteer_records():
        role = _ROLE_BY_LEVEL.get(str(row.get("level") or ""))
        if role and row.get("name") and row.get("code"):
            rows[role].append(row)
    return {role: tuple(values) for role, values in rows.items()}


@lru_cache(maxsize=1)
def _rows_by_name() -> dict[tuple[str, str], tuple[dict, ...]]:
    index: dict[tuple[str, str], list[dict]] = {}
    for role, rows in _rows_by_role().items():
        for row in rows:
            index.setdefault((role, _fold(row["name"])), []).append(row)
    return {key: tuple(values) for key, values in index.items()}


def _row_path(row: dict) -> dict[str, str]:
    path = {}
    for ancestor in ancestor_chain(str(row["code"])):
        role = _ROLE_BY_LEVEL.get(str(ancestor.get("level") or ""))
        if role:
            path[role] = str(ancestor["code"])
    return path


def _candidate_rows(
    value: str,
    roles: tuple[str, ...],
    allowed_units: set[str] | None = None,
) -> list[tuple[dict, bool]]:
    exact = [
        row for role in roles for row in _rows_by_name().get((role, value), ())
        if allowed_units is None or row.get("unit") in allowed_units
    ]
    if exact:
        return [(row, False) for row in exact]
    return []


def _fuzzy_candidate_rows(
    value: str,
    role: str,
    parent_codes: dict[str, set[str]],
    allowed_units: set[str] | None = None,
) -> list[tuple[dict, bool]]:
    """Conservative OCR fallback, constrained to an explicit parent when possible."""
    if len(value) < 5:
        return []
    if role == "district" and not parent_codes.get("province"):
        return []
    if role == "commune" and not (parent_codes.get("district") or parent_codes.get("province")):
        return []

    candidates = []
    for row in _rows_by_role()[role]:
        if allowed_units is not None and row.get("unit") not in allowed_units:
            continue
        canonical_name = _fold(row["name"])
        # A place name with an extra/missing trailing token is more likely a
        # boundary/concatenation error than a safe OCR substitution.
        if value.startswith(canonical_name) or canonical_name.startswith(value):
            continue
        path = _row_path(row)
        if any(codes and path.get(parent_role) not in codes
               for parent_role, codes in parent_codes.items()
               if parent_role != role):
            continue
        score = SequenceMatcher(None, value, canonical_name, autojunk=False).ratio()
        if score >= 0.86:
            candidates.append((score, row))
    candidates.sort(key=lambda item: (-item[0], str(item[1]["code"])))
    if not candidates:
        return []
    best_score = candidates[0][0]
    best = [row for score, row in candidates if score == best_score]
    next_score = next((score for score, _ in candidates if score < best_score), 0.0)
    # Never auto-correct a tied/near-tied place name.
    if len(best) != 1 or best_score - next_score < 0.08:
        return []
    return [(best[0], True)]


def _segment_rows(segment: dict, role: str, parent_codes: dict[str, set[str]]) -> list[tuple[dict, bool]]:
    prefix_roles = _PREFIX_LEVELS[segment["prefix"]]
    if role not in prefix_roles:
        return []
    return _resolve_segment_rows(segment, (role,), parent_codes)


def _resolve_segment_rows(
    segment: dict,
    roles: tuple[str, ...],
    parent_codes: dict[str, set[str]],
) -> list[tuple[dict, bool]]:
    """Resolve the longest official-name prefix, then try conservative OCR."""
    allowed_units = _PREFIX_UNITS.get(segment["prefix"])
    words = segment["name"].split()
    for stop in range(len(words), 0, -1):
        value = " ".join(words[:stop])
        exact = _candidate_rows(value, roles, allowed_units)
        compatible = []
        for row, fuzzy in exact:
            row_role = _ROLE_BY_LEVEL[str(row["level"])]
            compatible.extend(_filter_by_parents([(row, fuzzy)], parent_codes, row_role))
        if compatible:
            return compatible

    for stop in range(len(words), 0, -1):
        value = " ".join(words[:stop])
        fuzzy_rows = []
        for role in roles:
            for row, fuzzy in _fuzzy_candidate_rows(value, role, parent_codes, allowed_units):
                fuzzy_rows.extend(_filter_by_parents([(row, fuzzy)], parent_codes, role))
        if fuzzy_rows:
            return fuzzy_rows
    return []


def _lookup_name(segment: dict) -> str:
    """Expand only the common, explicitly unit-scoped TP.HCM abbreviation."""
    if segment["prefix"] in {"thanh pho", "tp"} and segment["name"] in {"hcm", "tp hcm"}:
        return "ho chi minh"
    return segment["name"]


def _filter_by_parents(
    candidates: list[tuple[dict, bool]], parent_codes: dict[str, set[str]], role: str
) -> list[tuple[dict, bool]]:
    filtered = []
    for row, fuzzy in candidates:
        path = _row_path(row)
        compatible = True
        for parent_role, codes in parent_codes.items():
            if parent_role == role or not codes:
                continue
            # Only administrative ancestors constrain this candidate.
            if parent_role in path and path[parent_role] not in codes:
                compatible = False
        if compatible:
            filtered.append((row, fuzzy))
    return filtered


def _parse_location(mention: dict) -> dict:
    if str(mention.get("label") or "").upper() != "LOC":
        return {}
    marked = _marked_clause(mention)
    if marked is None:
        return {}
    folded, target_span, target_surface = marked
    prefixes = list(_PREFIX_RE.finditer(folded))
    segments = []
    for index, match in enumerate(prefixes):
        stop = prefixes[index + 1].start() if index + 1 < len(prefixes) else len(folded)
        boundary = _COMPONENT_BOUNDARY.search(folded, match.end(), stop)
        if boundary:
            stop = boundary.start()
        name = _without_markers(folded[match.end():stop]).strip(" :-\t")
        if name:
            prefix = re.sub(r"\s+", " ", match.group("prefix").casefold())
            segments.append({
                "prefix": prefix,
                "roles": _PREFIX_LEVELS[prefix],
                "name": name,
                "start": match.start(),
                "end": stop,
                "candidates": [],
                "row": None,
                "fuzzy": False,
            })

    # Resolve from broad administrative units downward. A resolved parent
    # narrows repeated district/commune names; absent parents never become a
    # guessed conflict.
    parent_codes: dict[str, set[str]] = {role: set() for role in _LEVEL_BY_ROLE}
    explicit_groups = (
        ("province", {"tinh"}),
        ("district", {"huyen", "quan", "thi xa"}),
        ("commune", {"xa", "phuong", "thi tran"}),
    )
    for role, prefixes_for_role in explicit_groups:
        for segment in segments:
            if segment["prefix"] not in prefixes_for_role:
                continue
            candidates = _segment_rows(segment, role, parent_codes)
            candidates = _filter_by_parents(candidates, parent_codes, role)
            if len(candidates) == 1:
                segment["row"], segment["fuzzy"] = candidates[0]
                parent_codes[role].add(str(segment["row"]["code"]))

    # "Thành phố" is an official unit at both province and district levels.
    # Resolve across both levels at once; never silently prefer the broader one.
    for segment in segments:
        if segment["prefix"] not in {"thanh pho", "tp"} or segment["row"]:
            continue
        lookup_name = _lookup_name(segment)
        candidates = _resolve_segment_rows(
            {**segment, "name": lookup_name}, ("province", "district"), parent_codes
        )
        if len(candidates) == 1:
            segment["row"], segment["fuzzy"] = candidates[0]
            role = _ROLE_BY_LEVEL[str(segment["row"]["level"])]
            parent_codes[role].add(str(segment["row"]["code"]))

    # Resolve lower-level hamlet/village names as local aliases. They have no
    # gazetteer IDs in this resource, so they are only linkable under a shared
    # known commune/district parent and are never fuzzy-corrected.
    target_start, target_end = target_span
    target_segments = [segment for segment in segments
                       if segment["start"] < target_end and target_start < segment["end"]]
    target = None
    if len(target_segments) == 1:
        segment = target_segments[0]
        row = segment["row"]
        target = {
            "level": next((role for role in segment["roles"]
                           if row and _ROLE_BY_LEVEL.get(str(row["level"])) == role),
                          segment["roles"][0] if len(segment["roles"]) == 1 else "ambiguous"),
            "name": segment["name"],
            "code": str(row["code"]) if row else None,
            "fuzzy": bool(segment["fuzzy"]),
            "explicit_level": True,
            "unit_prefix": segment["prefix"] if not row else None,
        }
    elif len(target_segments) > 1:
        target = {
            "level": "address",
            "name": "|".join(f"{item['prefix']}:{item['name']}" for item in target_segments),
            "code": None,
            "fuzzy": any(item["fuzzy"] for item in target_segments),
            "explicit_level": True,
            "key": tuple(sorted(
                (next((role for role in item["roles"]
                       if item["row"] and _ROLE_BY_LEVEL.get(str(item["row"]["level"])) == role),
                      item["roles"][0]),
                 str(item["row"]["code"]) if item["row"] else item["name"])
                for item in target_segments
            )),
        }
    else:
        # NER may return only the place name and omit its unit prefix. Resolve
        # only a globally unique, non-marker-like gazetteer name; short
        # anonymization aliases (for example V, T2, or XX) must not resolve to
        # an unrelated real ward that happens to have the same spelling.
        if len(target_surface.replace(" ", "")) >= 3:
            candidates = [row for role in _LEVEL_BY_ROLE
                          for row in _rows_by_name().get((role, target_surface), ())]
            if len(candidates) == 1:
                row = candidates[0]
                target = {
                    "level": _ROLE_BY_LEVEL[str(row["level"])],
                    "name": target_surface,
                    "code": str(row["code"]),
                    "fuzzy": False,
                    "explicit_level": False,
                }
        if target is None and target_surface:
            # Use the nearest address-unit cue as a low-confidence type hint.
            preceding = [segment for segment in segments if segment["end"] <= target_start]
            if preceding:
                segment = preceding[-1]
                target = {"level": segment["roles"][0], "name": target_surface,
                      "code": None, "fuzzy": False, "explicit_level": False,
                      "unit_prefix": segment["prefix"]}

    path: dict[str, set[str]] = {role: set() for role in _LEVEL_BY_ROLE}
    for segment in segments:
        if segment["row"]:
            for role, code in _row_path(segment["row"]).items():
                path[role].add(code)
    for role, code in list(parent_codes.items()):
        path[role].update(code)

    # A masked district/city name can still be canonicalized by a recognized
    # child address in the same clause (e.g. "thị xã L, tỉnh X, xã Tân Bình")
    # when that child has exactly one gazetteer parent. This also resolves the
    # province-vs-district ambiguity of "thành phố" without guessing from the
    # surface alone.
    if target and not target.get("code"):
        level = target.get("level")
        if (level == "ambiguous" and target.get("unit_prefix") in {"thanh pho", "tp"}
                and len(path["district"]) == 1):
            target["level"] = "district"
            target["code"] = next(iter(path["district"]))
            target["inferred_from_hierarchy"] = True
        else:
            role = {"province": "province", "district": "district", "commune": "commune"}.get(level)
            if role and len(path[role]) == 1:
                target["code"] = next(iter(path[role]))
                target["inferred_from_hierarchy"] = True

    return {
        "target": target,
        "path": path,
        "ambiguous_path": any(len(codes) > 1 for codes in path.values()),
        "segments": segments,
    }


@lru_cache(maxsize=65536)
def _cached_location(label: str, context: str, surface: str) -> dict:
    return _parse_location({"label": label, "context": context, "surface": surface})


def location_signature(mention: dict) -> dict:
    """Expose parsed hierarchy without changing the NER-provided label."""
    return _cached_location(
        str(mention.get("label") or "").upper(),
        str(mention.get("context") or ""),
        str(mention.get("surface") or ""),
    )


def _shared_parent(left: dict, right: dict,
                   roles: tuple[str, ...] = ("commune", "district")) -> bool:
    left_path, right_path = left.get("path") or {}, right.get("path") or {}
    # Province alone is too broad to identify a hamlet or other local alias.
    return any(left_path.get(role) and left_path[role] == right_path.get(role)
               for role in roles)


_TARGET_PATH_ROLES = {
    "province": ("province",),
    "district": ("province", "district"),
    "commune": ("province", "district", "commune"),
    "hamlet": ("province", "district", "commune"),
    "address": ("province", "district", "commune"),
}


def _confident_level(target: dict) -> bool:
    return bool(
        target.get("level") in _TARGET_PATH_ROLES
        and (target.get("explicit_level") or target.get("code"))
    )


def _conflicting_target_levels(left: dict, right: dict) -> bool:
    target_a, target_b = left.get("target") or {}, right.get("target") or {}
    level_a, level_b = target_a.get("level"), target_b.get("level")
    return bool(
        level_a and level_b and level_a != level_b
        and _confident_level(target_a) and _confident_level(target_b)
    )


def _path_conflict(left: dict, right: dict) -> bool:
    left_path, right_path = left.get("path") or {}, right.get("path") or {}
    target_a, target_b = left.get("target") or {}, right.get("target") or {}
    level_a, level_b = target_a.get("level"), target_b.get("level")
    if level_a == level_b and level_a in _TARGET_PATH_ROLES:
        roles = _TARGET_PATH_ROLES[level_a]
    else:
        # Different/uncertain target types do not make child communes
        # contradictory (a district can contain many communes). Province is
        # the only safe comparison until the target levels agree.
        roles = ("province",)
    return any(left_path.get(role) and right_path.get(role)
               and left_path[role] != right_path[role]
               for role in roles)


def location_rule_decision(pair: dict) -> str | None:
    """Return ``link``, ``block``, or None (defer to the learned scorer)."""
    a, b = pair.get("mention_a") or {}, pair.get("mention_b") or {}
    if str(a.get("label") or "").upper() != "LOC" or str(b.get("label") or "").upper() != "LOC":
        return None
    left, right = location_signature(a), location_signature(b)
    if left.get("ambiguous_path") or right.get("ambiguous_path"):
        return None
    if _conflicting_target_levels(left, right):
        return "block"
    if _path_conflict(left, right):
        return "block"
    target_a, target_b = left.get("target") or {}, right.get("target") or {}
    level_a, level_b = target_a.get("level"), target_b.get("level")
    code_a, code_b = target_a.get("code"), target_b.get("code")
    if code_a and code_b:
        return "link" if code_a == code_b and level_a == level_b else "block"
    if (level_a == level_b == "address" and target_a.get("key")
            and target_a.get("key") == target_b.get("key")):
        return "link"
    if (level_a == level_b == "hamlet" and target_a.get("name") == target_b.get("name")
            and target_a.get("unit_prefix") == target_b.get("unit_prefix")
            and len(str(target_a.get("name") or "")) >= 3
            and _shared_parent(left, right, ("commune",))):
        return "link"
    return None


def location_pair_features(a: dict, b: dict) -> dict[str, float]:
    """Hierarchical match features; ORG/other labels get no location signals."""
    names = {
        "same_admin_unit": 0.0,
        "same_admin_level": 0.0,
        "same_admin_name_level": 0.0,
        "same_admin_parent": 0.0,
        "conflicting_admin_path": 0.0,
        "fuzzy_admin_match": 0.0,
    }
    if str(a.get("label") or "").upper() != "LOC" or str(b.get("label") or "").upper() != "LOC":
        return names
    left, right = location_signature(a), location_signature(b)
    target_a, target_b = left.get("target") or {}, right.get("target") or {}
    level_a, level_b = target_a.get("level"), target_b.get("level")
    code_a, code_b = target_a.get("code"), target_b.get("code")
    same_level = bool(level_a and level_a == level_b)
    same_unit = bool(same_level and code_a and code_a == code_b)
    if level_a == level_b == "address":
        same_unit = bool(target_a.get("key") and target_a.get("key") == target_b.get("key"))
    names["same_admin_unit"] = float(same_unit)
    names["same_admin_level"] = float(same_level)
    names["same_admin_name_level"] = float(
        same_level and bool(target_a.get("name")) and target_a.get("name") == target_b.get("name")
        and (level_a != "hamlet" or target_a.get("unit_prefix") == target_b.get("unit_prefix"))
    )
    names["same_admin_parent"] = float(_shared_parent(left, right))
    names["conflicting_admin_path"] = float(
        _path_conflict(left, right)
        or _conflicting_target_levels(left, right)
        or (same_level and bool(code_a and code_b and code_a != code_b))
    )
    names["fuzzy_admin_match"] = float(same_unit and (target_a.get("fuzzy") or target_b.get("fuzzy")))
    return names


def explicit_province(mention: dict) -> str | None:
    """Return the canonical province name only when the local path is unique."""
    # Keep the legacy convenience helper usable with contexts that omit NER
    # labels; automatic pair rules still gate on the actual LOC labels.
    typed_mention = mention if str(mention.get("label") or "").upper() == "LOC" else {
        **mention, "label": "LOC"
    }
    signature = location_signature(typed_mention)
    codes = (signature.get("path") or {}).get("province", set())
    if signature.get("ambiguous_path") or len(codes) != 1:
        return None
    code = next(iter(codes))
    for row in _rows_by_role()["province"]:
        if str(row["code"]) == code:
            return " ".join(str(row["name"]).casefold().translate(_DASHES).split())
    return None


def has_explicit_province_conflict(pair: dict) -> bool:
    """Backwards-compatible province-specific hard conflict check."""
    a, b = pair.get("mention_a") or {}, pair.get("mention_b") or {}
    if str(a.get("label") or "").upper() != "LOC" or str(b.get("label") or "").upper() != "LOC":
        return False
    left, right = location_signature(a), location_signature(b)
    provinces_a = (left.get("path") or {}).get("province", set())
    provinces_b = (right.get("path") or {}).get("province", set())
    return bool(
        not left.get("ambiguous_path") and not right.get("ambiguous_path")
        and len(provinces_a) == len(provinces_b) == 1
        and provinces_a != provinces_b
    )


def has_explicit_location_conflict(pair: dict) -> bool:
    return location_rule_decision(pair) == "block"


def location_province_features(a: dict, b: dict) -> dict[str, float]:
    """Legacy province feature names retained for checkpoint compatibility."""
    if str(a.get("label") or "").upper() != "LOC" or str(b.get("label") or "").upper() != "LOC":
        return {"same_explicit_province": 0.0, "conflicting_explicit_province": 0.0}
    province_a, province_b = explicit_province(a), explicit_province(b)
    return {
        "same_explicit_province": float(province_a is not None and province_a == province_b),
        "conflicting_explicit_province": float(
            province_a is not None and province_b is not None and province_a != province_b
        ),
    }
