"""Legal encoding plan for Vietnamese court judgments after entity linking.

Resolution 03/2017/NQ-HĐTP, Article 7 and Supreme People's Court guidance
144/TANDTC-PC require coding protected participant and organization names and
address details, while preserving proceeding personnel/agencies and province
or centrally administered city names. Address components are coded at their
own administrative level so a full address can retain its lawful structure.
"""

from __future__ import annotations

import re
import unicodedata


_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_ORG_PREFIXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"^công\s+ty\s+trách\s+nhiệm\s+hữu\s+hạn\b",
    r"^công\s+ty\s+tnhh\b",
    r"^công\s+ty\s+cổ\s+phần\b",
    r"^công\s+ty\s+cp\b",
    r"^công\s+ty\s+hợp\s+danh\b",
    r"^doanh\s+nghiệp\s+tư\s+nhân\b",
    r"^hợp\s+tác\s+xã\b",
    r"^ủy\s+ban\s+nhân\s+dân\b",
    r"^ubnd\b",
    r"^ngân\s+hàng\b",
    r"^trường\s+đại\s+học\b",
    r"^trường\b",
    r"^bệnh\s+viện\b",
    r"^tập\s+đoàn\b",
    r"^trung\s+tâm\b",
    r"^chi\s+cục\b",
    r"^cục\b",
    r"^sở\b",
    r"^phòng\b",
    r"^ban\b",
    r"^viện\b",
    r"^đài\b",
    r"^báo\b",
    r"^nhà\s+máy\b",
    r"^cửa\s+hàng\b",
    r"^cơ\s+quan\b",
))
_ADMIN_UNIT = re.compile(
    r"^\s*(tỉnh|thành\s+phố|quận|huyện|thị\s+xã|xã|phường)\b", re.IGNORECASE
)
_ADMIN_TAIL = re.compile(
    r"[,;]\s*(?:tỉnh|thành\s+phố|tp\.?)\s+[^,;]+\s*$", re.IGNORECASE
)
_ADDRESS_NUMBER = re.compile(
    r"\bsố\s+\d+[A-Z0-9/-]*\s*(?=,?\s*(?:đường|phố|ngõ|hẻm|kiệt)\b)",
    re.IGNORECASE,
)
_ADDRESS_COMPONENT = re.compile(
    r"(?<![\w])(?P<prefix>tổ\s+dân\s+phố|cụm\s+dân\s+cư|khu\s+dân\s+cư|"
    r"thành\s+phố|thị\s+trấn|thị\s+xã|khu\s+phố|tỉnh|huyện|quận|xã|"
    r"phường|ấp|thôn|xóm|bản|buôn|bon|làng|khóm|phum|sóc|khu|tổ|tp\.?|"
    r"đường|phố|ngõ|hẻm|kiệt)(?![\w])\s*[:,\-]?\s*",
    re.IGNORECASE,
)
_LOCATION_CODES = tuple("MNPQRSTUVWXYZABCDEFGHIJKLO")
_ORGANIZATION_CODES = tuple("BCDEFGHIJKLMNOPQRSTUVWXYZA")
_LEVELS = {"province", "district", "commune", "hamlet", "street", "address"}


def _fold(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    return "".join(
        char for char in unicodedata.normalize("NFD", value.casefold())
        if not unicodedata.combining(char)
    )


def _initial(value: str) -> str:
    first = next((char for char in value if char.isalpha()), "")
    if not first:
        return ""
    if first.casefold() == "đ":
        return "Đ"
    decomposed = unicodedata.normalize("NFD", first)
    return next((char.upper() for char in decomposed if not unicodedata.combining(char)), "")


def _valid_span(text: str, mention: dict) -> tuple[int, int] | None:
    try:
        start, end = int(mention["start"]), int(mention["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= start < end <= len(text):
        return None
    supplied = mention.get("surface")
    if supplied is not None and str(supplied) != text[start:end]:
        return None
    return start, end


def _mentions(text: str, cluster: dict, label: str) -> list[dict]:
    found = []
    for mention in cluster.get("mentions") or []:
        if str(mention.get("label") or label).upper() != label:
            continue
        span = _valid_span(text, mention)
        if span is None:
            continue
        start, end = span
        surface = text[start:end]
        tokens = list(_WORD_RE.finditer(surface))
        if tokens:
            found.append({**mention, "start": start, "end": end,
                          "surface": surface, "tokens": tokens})
    return found


def _organization_parts(surface: str) -> tuple[str, str, str]:
    """Return legal form, identifying name, and public administrative tail."""
    tail_match = _ADMIN_TAIL.search(surface)
    tail = surface[tail_match.start():] if tail_match else ""
    core = surface[:tail_match.start()].rstrip() if tail_match else surface
    for pattern in _ORG_PREFIXES:
        match = pattern.match(core)
        if not match:
            continue
        prefix = core[:match.end()].rstrip()
        remainder = core[match.end():]
        # Keep the public body/unit form and mask its identifying locality:
        # UBND quận Hoàn Kiếm -> UBND quận H.
        if _fold(match.group()) in {"ubnd", "uy ban nhan dan"}:
            unit = _ADMIN_UNIT.match(remainder)
            if unit:
                prefix = core[:match.end() + unit.end()].rstrip()
                remainder = core[match.end() + unit.end():]
        return prefix, remainder.strip(), tail
    return "", core.strip(), tail


def _location_signature(mention: dict) -> dict:
    # Reuse the linker's hierarchy and gazetteer instead of guessing when
    # "thành phố" could refer to either a province-level or district-level city.
    from ..entity_linking.location_constraints import location_signature

    return location_signature(mention)


def _location_level(mention: dict) -> str | None:
    target = _location_signature(mention).get("target") or {}
    level = str(target.get("level") or "")
    return level if level in _LEVELS or level == "ambiguous" else None


def _component_spans(surface: str, absolute_start: int) -> list[dict]:
    matches = list(_ADDRESS_COMPONENT.finditer(surface))
    components = []
    for index, match in enumerate(matches):
        limit = matches[index + 1].start() if index + 1 < len(matches) else len(surface)
        delimiter = re.search(r"[,.;!?\r\n]+", surface[match.end():limit])
        end = match.end() + delimiter.start() if delimiter else limit
        while end > match.start() and surface[end - 1].isspace():
            end -= 1
        while end > match.end() and surface[end - 1] in ",.;:!?-":
            end -= 1
        if end <= match.end():
            continue
        components.append({
            "start": absolute_start + match.start(),
            "end": absolute_start + end,
            "surface": surface[match.start():end],
            "prefix": match.group("prefix"),
        })
    return components


def _component_mention(text: str, component: dict) -> dict:
    # Wrap only this address unit, while retaining adjacent context so the
    # location resolver can distinguish province cities from district cities.
    from ..entity_linking.dataset import MENTION_CLOSE, MENTION_OPEN

    start, end = component["start"], component["end"]
    left, right = max(0, start - 180), min(len(text), end + 180)
    context = text[left:start] + MENTION_OPEN + text[start:end] + MENTION_CLOSE + text[end:right]
    return {"label": "LOC", "start": start, "end": end,
            "surface": text[start:end], "context": context}


def _explicit_level(prefix: str) -> str | None:
    prefix = _fold(prefix)
    if prefix == "tinh":
        return "province"
    if prefix in {"huyen", "quan", "thi xa"}:
        return "district"
    if prefix in {"xa", "phuong", "thi tran"}:
        return "commune"
    if prefix in {"ap", "thon", "xom", "ban", "buon", "bon", "lang", "khom",
                  "phum", "soc", "khu pho", "khu dan cu", "cum dan cu", "to dan pho",
                  "to", "khu"}:
        return "hamlet"
    if prefix in {"duong", "pho", "ngo", "hem", "kiet"}:
        return "street"
    return None


def _component_key(component: dict, level: str, signature: dict) -> tuple:
    target = signature.get("target") or {}
    path = signature.get("path") or {}
    path_key = tuple((role, tuple(sorted(map(str, values))))
                     for role, values in sorted(path.items()) if values)
    code = str(target.get("code") or "")
    name = _fold(str(target.get("name") or component["surface"]))
    return level, code or name, path_key if not code else ()


def _org_replacement(surface: str, code: str, location_code: str | None = None) -> str:
    prefix, _name, tail = _organization_parts(surface)
    replacement_code = location_code or code
    encoded = f"{prefix} {replacement_code}".strip() if prefix else replacement_code
    return encoded + tail


def _organization_location(text: str, mention: dict) -> dict | None:
    """Find a lower-level locality embedded in an UBND name.

    For example, in ``Ủy ban nhân dân thị xã Điện Bàn, tỉnh Quảng Nam``,
    the legal body and administrative unit are retained while the locality
    name is encoded with the same location code used for standalone mentions.
    Province-level units are outside this lower-level locality path and stay
    under the existing organization/location policy.
    """
    surface = mention["surface"]
    prefix, name, _tail = _organization_parts(surface)
    if not prefix or not name or not re.match(
        r"^\s*(?:ủy\s+ban\s+nhân\s+dân|ubnd)\b", prefix, re.IGNORECASE
    ):
        return None

    body = re.match(r"^\s*(?:ủy\s+ban\s+nhân\s+dân|ubnd)\b", prefix, re.IGNORECASE)
    unit_offset = body.end() if body else 0
    unit_match = _ADMIN_UNIT.match(prefix[unit_offset:])
    if unit_match is None:
        return None

    name_start = surface.find(name, len(prefix))
    if name_start < 0:
        return None
    unit_start = unit_offset + unit_match.start()
    absolute_start = mention["start"] + unit_start
    absolute_end = mention["start"] + name_start + len(name)
    component = {
        "start": absolute_start,
        "end": absolute_end,
        "surface": text[absolute_start:absolute_end],
        "prefix": unit_match.group(1),
    }
    synthetic_mention = _component_mention(text, component)
    signature = _location_signature(synthetic_mention)
    target = signature.get("target") or {}
    level = _explicit_level(component["prefix"]) or str(target.get("level") or "")
    if _fold(component["prefix"]) in {"thanh pho", "tp"}:
        level = str(target.get("level") or "ambiguous")
    if level == "province" or level not in {"district", "commune", "hamlet", "street", "address"}:
        return None

    key = _component_key(component, level, signature)
    return {
        "key": key,
        "start": absolute_start,
        "end": absolute_end,
        "name_start": mention["start"] + name_start,
        "name_end": mention["start"] + name_start + len(name),
        "level": level,
        "component": component,
        "signature": signature,
    }


def _candidate_replacement(candidate: dict, mention: dict, code: str) -> str:
    if candidate["label"] == "PER":
        return _person_replacement(mention, code)
    location = candidate.get("organization_locations", {}).get(mention["start"])
    return _org_replacement(mention["surface"], code,
                            location_code=location["code"] if location else None)


def _organization_masked_name_span(surface: str, absolute_start: int) -> tuple[int, int] | None:
    """Return the identifying name span replaced by the ORG code."""
    prefix, name, _tail = _organization_parts(surface)
    name_start = surface.find(name, len(prefix)) if name else -1
    if name_start < 0:
        return None
    return absolute_start + name_start, absolute_start + name_start + len(name)


def _person_replacement(mention: dict, code: str) -> str:
    surface = mention["surface"]
    tokens = mention["tokens"]
    token = tokens[-1] if len(tokens) > 1 else tokens[0]
    return surface[:token.start()] + code + surface[token.end():]


def _render_surface(surface: str, replacements: list[dict], absolute_start: int) -> str:
    result = surface
    for item in sorted(replacements, key=lambda row: row["start"], reverse=True):
        start, end = item["start"] - absolute_start, item["end"] - absolute_start
        result = result[:start] + item["replacement"] + result[end:]
    return result


def _next_location_code(index: int) -> str:
    base = _LOCATION_CODES[index % len(_LOCATION_CODES)]
    return base if index < len(_LOCATION_CODES) else f"{base}{index // len(_LOCATION_CODES) + 1}"


def _next_organization_code(index: int) -> str:
    base = _ORGANIZATION_CODES[index % len(_ORGANIZATION_CODES)]
    return base if index < len(_ORGANIZATION_CODES) else f"{base}{index // len(_ORGANIZATION_CODES) + 1}"


def build_anonymization_plan(text: str, clusters: list[dict]) -> dict:
    """Plan linked PER/ORG/LOC replacements under the court publication rule.

    People retain the initial of the final name token, with a numeric suffix
    only when distinct linked groups would otherwise produce the same visible
    name. Organizations receive document-local aliases independent of their
    names. Lower-level address units each receive a stable letter code;
    province-level locations remain visible. Unresolved city levels and
    invalid or overlapping spans are sent for review rather than guessed.
    """
    entities: list[dict] = []
    candidates: list[dict] = []
    location_entries: list[dict] = []
    replacements: list[dict] = []
    location_codes: dict[tuple, str] = {}
    supported = {"PER", "ORG", "LOC"}

    for index, cluster in enumerate(clusters, start=1):
        label = str(cluster.get("label") or "").upper()
        if label not in supported:
            continue
        entity_id = str(cluster.get("cluster_id") or cluster.get("entity_id") or f"{label}_{index:04d}")
        if cluster.get("anonymization_action") == "preserve":
            entities.append({"entity_id": entity_id, "label": label, "action": "preserve",
                             "reason": cluster.get("anonymization_reason") or "protected_by_policy",
                             "code": None, "codes": [], "encoded_name": None})
            continue
        mentions = _mentions(text, cluster, label)
        if not mentions:
            reason = "entity_name_span_missing_or_invalid"
            cluster.update(anonymization_action="review", anonymization_reason=reason)
            entities.append({"entity_id": entity_id, "label": label, "action": "review",
                             "reason": reason, "code": None, "codes": [], "encoded_name": None})
            continue

        canonical = max(mentions, key=lambda item: (len(item["tokens"]), len(item["surface"])))
        if label in {"PER", "ORG"}:
            if label == "PER":
                code_base = _initial(canonical["tokens"][-1].group())
            else:
                code_base = ""
            if label == "PER" and not code_base:
                reason = "entity_encoding_initial_unavailable"
                cluster.update(anonymization_action="review", anonymization_reason=reason)
                entities.append({"entity_id": entity_id, "label": label, "action": "review",
                                 "reason": reason, "code": None, "codes": [], "encoded_name": None})
                continue
            candidates.append({"entity_id": entity_id, "cluster": cluster, "label": label,
                               "mentions": mentions, "canonical": canonical,
                               "code_base": code_base,
                               "first_start": min(item["start"] for item in mentions)})
            continue

        # LOC spans can contain an entire address. Split them into explicit
        # address units so each protected level gets its own code while the
        # province/city at province level remains intact.
        mention_parts = []
        review_reason = None
        any_masked = False
        for mention in mentions:
            components = _component_spans(mention["surface"], mention["start"])
            parts = []
            if components:
                for component in components:
                    component_mention = _component_mention(text, component)
                    signature = _location_signature(component_mention)
                    target = signature.get("target") or {}
                    level = _explicit_level(component["prefix"]) or str(target.get("level") or "")
                    if _fold(component["prefix"]) in {"thanh pho", "tp"}:
                        level = str(target.get("level") or "ambiguous")
                    if level == "province":
                        continue
                    if level not in {"district", "commune", "hamlet", "street", "address"}:
                        review_reason = "location_level_ambiguous_or_composite_address"
                        break
                    key = _component_key(component, level, signature)
                    code = location_codes.get(key)
                    if code is None:
                        code = _next_location_code(len(location_codes))
                        location_codes[key] = code
                    prefix = component["prefix"].strip()
                    parts.append({"entity_id": entity_id, "label": label,
                                  "start": component["start"], "end": component["end"],
                                  "original": text[component["start"]:component["end"]],
                                  "replacement": f"{prefix} {code}", "code": code,
                                  "_location_key": key})
                    any_masked = True
            else:
                signature = _location_signature(mention)
                level = str((signature.get("target") or {}).get("level") or "")
                if level == "province":
                    parts = []
                elif level in {"district", "commune", "hamlet", "street"}:
                    component = {"surface": mention["surface"], "prefix": ""}
                    key = _component_key(component, level, signature)
                    code = location_codes.get(key)
                    if code is None:
                        code = _next_location_code(len(location_codes))
                        location_codes[key] = code
                    parts.append({"entity_id": entity_id, "label": label,
                                  "start": mention["start"], "end": mention["end"],
                                  "original": mention["surface"], "replacement": code, "code": code,
                                  "_location_key": key})
                    any_masked = True
                else:
                    review_reason = "location_level_ambiguous_or_composite_address"
            if review_reason:
                break
            entry_replacements = parts
            mention["planned_anonymized_name"] = _render_surface(
                mention["surface"], entry_replacements, mention["start"]
            )
            mention_parts.append({"mention": mention, "replacements": parts})

        if review_reason:
            cluster.update(anonymization_action="review", anonymization_reason=review_reason)
            entities.append({"entity_id": entity_id, "label": label, "action": "review",
                             "reason": review_reason, "code": None, "codes": [], "encoded_name": None})
        elif not any_masked:
            reason = "administrative_province_or_centrally_administered_city"
            cluster.update(anonymization_action="preserve", anonymization_reason=reason)
            entities.append({"entity_id": entity_id, "label": label, "action": "preserve",
                             "reason": reason, "code": None, "codes": [], "encoded_name": None})
        else:
            codes = []
            for entry in mention_parts:
                for item in entry["replacements"]:
                    if item["code"] not in codes:
                        codes.append(item["code"])
                    replacement_item = {key: value for key, value in item.items() if key != "code"}
                    replacement_item["_location_code"] = item["code"]
                    replacements.append(replacement_item)
            canonical_entry = max(
                (entry for entry in mention_parts if entry["replacements"]),
                key=lambda entry: (len(entry["mention"]["surface"]), entry["mention"]["start"]),
            )
            encoded_name = canonical_entry["mention"]["planned_anonymized_name"]
            cluster.update(anonymization_action="anonymize", anonymization_code=codes[0] if len(codes) == 1 else None,
                           anonymized_name=encoded_name)
            location_entries.append({"entity_id": entity_id, "label": label, "cluster": cluster,
                                     "codes": codes, "encoded_name": encoded_name})

    # A public-body ORG can contain a lower-level locality in its own name.
    # Give that locality the same code as a matching LOC mention elsewhere.
    # This also lets the outer ORG span absorb a nested LOC edit without an
    # overlapping replacement.
    for candidate in candidates:
        if candidate["label"] != "ORG":
            continue
        organization_locations = {}
        for mention in candidate["mentions"]:
            location = _organization_location(text, mention)
            if location is None:
                continue
            nested_locations = [
                item for item in replacements
                if item["label"] == "LOC"
                and location["start"] <= item["start"]
                and item["end"] <= location["end"]
                and item.get("_location_code")
            ]
            code = location_codes.get(location["key"])
            if nested_locations:
                # Prefer the code already planned for the matching inner LOC.
                matching = next((item for item in nested_locations
                                 if item.get("_location_key") == location["key"]), nested_locations[0])
                code = matching["_location_code"]
                location_codes[location["key"]] = code
            elif code is None:
                code = _next_location_code(len(location_codes))
                location_codes[location["key"]] = code
            location["code"] = code
            organization_locations[mention["start"]] = location
        candidate["organization_locations"] = organization_locations

    # Organization aliases are document-local codes, not initials from the
    # organization name. Start at B to match the illustrative company code
    # in Resolution 03/2017/NQ-HĐTP; the sequence only supplies distinct IDs.
    organization_candidates = sorted(
        (candidate for candidate in candidates if candidate["label"] == "ORG"),
        key=lambda item: (item["first_start"], item["entity_id"]),
    )
    for ordinal, candidate in enumerate(organization_candidates):
        candidate["code_base"] = _next_organization_code(ordinal)

    # If a public-body name contains a lower-level locality, that locality's
    # code takes precedence over the generic organization alias.
    for candidate in organization_candidates:
        canonical_location = candidate.get("organization_locations", {}).get(
            candidate["canonical"]["start"]
        )
        if canonical_location:
            candidate["code_base"] = canonical_location["code"]

    # Add numbers only when two distinct entities would otherwise produce the
    # same visible anonymized mention. This catches collisions between short
    # aliases (for example, two different people both mentioned as ``An``)
    # without turning every T-initial person or organization into T1...Tn.
    unnumbered_owners: dict[str, set[str]] = {}
    for candidate in candidates:
        for mention in candidate["mentions"]:
            visible = _candidate_replacement(candidate, mention, candidate["code_base"])
            key = " ".join(_fold(visible).split())
            unnumbered_owners.setdefault(key, set()).add(candidate["entity_id"])
    colliding_ids = {
        entity_id
        for owners in unnumbered_owners.values() if len(owners) > 1
        for entity_id in owners
    }
    candidate_by_id = {item["entity_id"]: item for item in candidates}
    parents = {entity_id: entity_id for entity_id in colliding_ids}

    def root(entity_id: str) -> str:
        while parents[entity_id] != entity_id:
            parents[entity_id] = parents[parents[entity_id]]
            entity_id = parents[entity_id]
        return entity_id

    def join(left_id: str, right_id: str) -> None:
        left_root, right_root = root(left_id), root(right_id)
        if left_root != right_root:
            parents[right_root] = left_root

    # Only connect groups that collide on an actual visible mention. Grouping
    # every entity with the same initial lets unrelated organization and person
    # collisions consume each other's numbers (the cause of inflated T codes).
    for owners in unnumbered_owners.values():
        colliding_owners = sorted(owners.intersection(colliding_ids))
        if len(colliding_owners) > 1:
            for entity_id in colliding_owners[1:]:
                join(colliding_owners[0], entity_id)

    collision_components: dict[str, list[dict]] = {}
    for entity_id in colliding_ids:
        collision_components.setdefault(root(entity_id), []).append(candidate_by_id[entity_id])
    assigned_codes = {}
    for group in collision_components.values():
        group.sort(key=lambda item: (item["first_start"], item["entity_id"]))
        used_codes = set()
        next_ordinal: dict[str, int] = {}
        for candidate in group:
            code_base = candidate["code_base"]
            ordinal = next_ordinal.get(code_base, 0) + 1
            code = f"{code_base}{ordinal}"
            while code in used_codes:
                ordinal += 1
                code = f"{code_base}{ordinal}"
            next_ordinal[code_base] = ordinal
            used_codes.add(code)
            assigned_codes[candidate["entity_id"]] = code

    for candidate in candidates:
        code = assigned_codes.get(candidate["entity_id"], candidate["code_base"])
        candidate["cluster"].update(anonymization_action="anonymize",
                                    anonymization_code=code)
        encoded_name = None
        for mention in candidate["mentions"]:
            replacement = _candidate_replacement(candidate, mention, code)
            item = {"entity_id": candidate["entity_id"], "label": candidate["label"],
                    "start": mention["start"], "end": mention["end"],
                    "original": mention["surface"], "replacement": replacement}
            if candidate["label"] == "ORG":
                masked_name_span = _organization_masked_name_span(mention["surface"], mention["start"])
                if masked_name_span:
                    item["_organization_masked_name_span"] = masked_name_span
                item["_organization_code"] = code
                location = candidate.get("organization_locations", {}).get(mention["start"])
                if location:
                    item["_organization_location_span"] = (location["start"], location["end"])
                    item["_organization_location_name_span"] = (location["name_start"], location["name_end"])
                    item["_organization_location_code"] = location["code"]
            replacements.append(item)
            mention["planned_anonymized_name"] = replacement
            if mention["start"] == candidate["canonical"]["start"]:
                encoded_name = replacement
        candidate["cluster"]["anonymized_name"] = encoded_name
        entities.append({"entity_id": candidate["entity_id"], "label": candidate["label"],
                         "action": "anonymize", "code": code, "codes": [code],
                         "encoded_name": encoded_name})

    for entry in location_entries:
        entities.append({"entity_id": entry["entity_id"], "label": entry["label"],
                         "action": "anonymize", "code": entry["codes"][0] if len(entry["codes"]) == 1 else None,
                         "codes": entry["codes"], "encoded_name": entry["encoded_name"]})

    # The legal address example drops a building number before an encoded
    # street. Remove such numbers unless a more specific entity span covers it.
    replacement_spans = [(item["start"], item["end"]) for item in replacements]
    for match in _ADDRESS_NUMBER.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in replacement_spans):
            continue
        replacements.append({"entity_id": f"ADDRESS_DETAIL_{match.start():08d}",
                             "label": "ADDRESS_DETAIL", "start": match.start(), "end": match.end(),
                             "original": text[match.start():match.end()], "replacement": ""})

    # Resolve nested NER spans locally. A longer ORG mention already encodes
    # its identifying name; a nested ORG fragment is a span artifact, and a
    # nested LOC in the UBND locality is encoded inside that outer replacement.
    org_replacements = [item for item in replacements if item["label"] == "ORG"]
    covered_replacements = set()
    candidate_by_id = {candidate["entity_id"]: candidate for candidate in candidates}
    for inner in replacements:
        containers = [
            outer for outer in org_replacements
            if outer is not inner
            and outer["start"] <= inner["start"]
            and inner["end"] <= outer["end"]
            and (outer["start"] < inner["start"] or inner["end"] < outer["end"])
        ]
        if not containers:
            continue
        # If ORG mentions are nested in several layers, map the inner span
        # against the outermost replacement that will determine final text.
        outer = max(containers, key=lambda item: item["end"] - item["start"])
        inner_label = inner["label"]
        masked_name = outer.get("_organization_masked_name_span")
        location_span = outer.get("_organization_location_span")
        covered = inner_label == "ORG"
        if inner_label == "LOC":
            covered = bool(
                (masked_name and masked_name[0] <= inner["start"] and inner["end"] <= masked_name[1])
                or (location_span and location_span[0] <= inner["start"] and inner["end"] <= location_span[1])
            )
        if not covered:
            continue

        covered_replacements.add(id(inner))
        inner_candidate = candidate_by_id.get(inner["entity_id"])
        if inner_candidate:
            for mention in inner_candidate["mentions"]:
                if mention["start"] != inner["start"] or mention["end"] != inner["end"]:
                    continue
                name_span = outer.get("_organization_location_name_span")
                replacement_code = outer.get("_organization_location_code")
                if not replacement_code:
                    replacement_code = outer.get("_organization_code")
                if name_span and replacement_code:
                    if inner["start"] <= name_span[0] and name_span[1] <= inner["end"]:
                        mention["planned_anonymized_name"] = _render_surface(
                            inner["original"],
                            [{"start": name_span[0], "end": name_span[1],
                              "replacement": replacement_code}],
                            inner["start"],
                        )
                    elif name_span[0] <= inner["start"] and inner["end"] <= name_span[1]:
                        mention["planned_anonymized_name"] = replacement_code
                    else:
                        mention["planned_anonymized_name"] = inner["original"]
                elif masked_name and masked_name[0] <= inner["start"] and inner["end"] <= masked_name[1]:
                    mention["planned_anonymized_name"] = outer.get("_organization_code")
                else:
                    mention["planned_anonymized_name"] = inner["original"]
                break

    replacements = [item for item in replacements if id(item) not in covered_replacements]
    ordered = sorted(replacements, key=lambda item: (item["start"], -item["end"]))
    conflicts = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            if right["start"] >= left["end"]:
                break
            conflicts.extend((left["entity_id"], right["entity_id"]))
    if conflicts:
        conflict_ids = sorted(set(conflicts))
        replacements = [item for item in replacements if item["entity_id"] not in conflict_ids]
        for cluster in clusters:
            identifier = str(cluster.get("cluster_id") or cluster.get("entity_id") or "")
            if identifier in conflict_ids:
                cluster.update(anonymization_action="review", anonymization_reason="overlapping_entity_spans")
        for entity in entities:
            if entity["entity_id"] in conflict_ids:
                entity.update(action="review", reason="overlapping_entity_spans",
                              code=None, codes=[], encoded_name=None)

    entities.sort(key=lambda item: item["entity_id"])
    review_ids = sorted(item["entity_id"] for item in entities if item["action"] == "review")
    public_replacements = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in sorted(replacements, key=lambda item: (item["start"], item["end"]))
    ]
    return {
        "algorithm": "vietnamese_court_legal_encoding",
        "legal_basis": "Resolution 03/2017/NQ-HĐTP Article 7; Official Letter 144/TANDTC-PC section 2",
        "scope": "PER, ORG, LOC and explicit street-number address details",
        "status": "needs_review" if conflicts or review_ids else "ready",
        "entities": entities,
        "replacements": public_replacements,
        "conflicting_entity_ids": sorted(set(conflicts)),
        "review_entity_ids": review_ids,
    }


def build_person_encoding_plan(text: str, clusters: list[dict]) -> dict:
    """Compatibility wrapper for older PER-only callers."""
    people = [cluster for cluster in clusters if str(cluster.get("label") or "").upper() == "PER"]
    return build_anonymization_plan(text, people)


def apply_span_replacements(text: str, replacements: list[dict]) -> str:
    """Apply validated, non-overlapping span edits from right to left."""
    ordered = sorted(replacements, key=lambda item: (int(item["start"]), int(item["end"])))
    previous_end = 0
    for item in ordered:
        start, end = int(item["start"]), int(item["end"])
        if not previous_end <= start < end <= len(text):
            raise ValueError("Invalid or overlapping replacement spans")
        if item.get("original") is not None and text[start:end] != item["original"]:
            raise ValueError(f"Source text mismatch at span [{start}, {end})")
        previous_end = end
    result = text
    for item in reversed(ordered):
        start, end = int(item["start"]), int(item["end"])
        result = result[:start] + str(item["replacement"]) + result[end:]
    return result
