"""Heuristic split of person/organisation mentions into court personnel versus
case participants.

Vietnamese court judgments use recurring role titles for judges, assessors,
clerks and prosecutors in composition lists and signature blocks. Party names
can also appear earlier in the ruling before a formal party heading, so this
heuristic requires a nearby procedural title instead of treating the entire
opening section as court personnel.

This module identifies proceeding personnel and agencies in an *un-anonymised*
document. It is wired into document-linker predictions so the decision is
recorded on the entity cluster and propagated to its linked mentions. It is
not used by the reconstruction pipeline in ``core.reconstruction``, which
fills already-masked markers to create synthetic training text.
"""
from __future__ import annotations

import re
import unicodedata

# Titles that only ever refer to the court itself, never to a party. Matched
# case-insensitively; Vietnamese judgments are typically upper-cased here but
# not reliably so.
COURT_ROLE_TITLES = re.compile(
    r"(THẨM PHÁN|HỘI THẨM NHÂN DÂN|CHỦ TỌA PHIÊN TÒA|"
    r"THƯ\s*KÝ(?:\s+PHIÊN TÒA|\s+TÒA ÁN)?|"
    r"TM\.\s*HỘI ĐỒNG XÉT XỬ|HỘI ĐỒNG XÉT XỬ|"
    r"CHÁNH ÁN|PHÓ CHÁNH ÁN|KIỂM SÁT VIÊN)",
    re.IGNORECASE,
)

# Closing-section markers bound local role-title searches. They identify the
# signature/distribution area, but do not by themselves classify every entity
# after them as court personnel.
FOOTER_START = re.compile(r"(TM\.\s*HỘI ĐỒNG XÉT XỬ|Nơi nhận\s*:)", re.IGNORECASE)

# Formal party labels provide a useful section boundary for local context
# searches. They do not imply that every name before the first label is court
# personnel; some decisions name the parties in the ruling first.
FIRST_PARTY_DECLARATION = re.compile(
    r"(nguyên đơn|bị đơn|bị cáo|bị hại|người làm chứng|người khởi kiện|người bị kiện)\s*:",
    re.IGNORECASE,
)

# Accent-folded patterns cover ordinary Unicode Vietnamese as well as OCR text
# with inconsistent capitalization. Folding is offset-preserving for the
# Vietnamese alphabet, so original NER character spans remain valid.
FOLDED_COURT_ROLE_TITLES = re.compile(
    r"\b(?:tham phan|hoi tham nhan dan|chu toa phien toa|"
    r"thu ky(?: phien toa| toa an)?|tm\.\s*hoi dong xet xu|"
    r"hoi dong xet xu|pho chanh an|chanh an|kiem sat vien|"
    r"dieu tra vien|can bo dieu tra)\b",
    re.IGNORECASE,
)
DIRECT_PROCEEDING_ROLE = re.compile(
    r"\b(?:tham phan|hoi tham nhan dan|thanh vien hoi dong xet xu|chu toa phien toa|"
    r"thu ky(?: phien toa| toa an)?|kiem sat vien|dieu tra vien|can bo dieu tra)\b",
    re.IGNORECASE,
)
PROCEEDING_AGENCY = re.compile(
    r"\b(?:tand(?:tc)?|toa an(?: nhan dan)?|"
    r"vksnd(?:tc)?|vien kiem sat(?: nhan dan)?|"
    r"co quan (?:canh sat|an ninh) dieu tra|co quan csdt|cqdt|csdt)\b",
    re.IGNORECASE,
)
FOLDED_FOOTER_START = re.compile(r"(?:tm\.\s*hoi dong xet xu|noi nhan\s*:)", re.IGNORECASE)
FOLDED_FIRST_PARTY_DECLARATION = re.compile(
    r"\b(?:nguyen don|bi don|bi cao|bi hai|bi hoi|nguoi lam chung|"
    r"nguoi khoi kien|nguoi bi kien)\s*:",
    re.IGNORECASE,
)


def _fold_vietnamese(text: str) -> str:
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(char)
    )


def signature_block_start(text: str) -> int | None:
    """Offset where the judgment's closing signature block begins, if present.

    Uses the LAST match: a document can legitimately repeat court-role
    language earlier (see COURT_ROLE_TITLES), and the footer is always the
    final such occurrence, never the first.
    """
    starts = [match.start() for match in FOOTER_START.finditer(text)]
    starts.extend(match.start() for match in FOLDED_FOOTER_START.finditer(_fold_vietnamese(text)))
    return max(starts) if starts else None


def header_zone_end(text: str) -> int:
    """Offset of the first formal party declaration, or 0 when absent.

    This is only a context-window boundary, not a classification rule.
    """
    starts = [match.start() for match in FIRST_PARTY_DECLARATION.finditer(text)]
    starts.extend(match.start() for match in FOLDED_FIRST_PARTY_DECLARATION.finditer(_fold_vietnamese(text)))
    return min(starts) if starts else 0


def is_court_personnel_context(text: str, start: int, end: int, window: int = 120) -> bool:
    """True when a procedural title structurally identifies this mention.

    Search windows stay inside the opening, body, or footer section. A name
    is never protected only because it appears before a party label or
    after the footer begins.
    """
    folded = _fold_vietnamese(text)
    direct_left = folded[max(0, start - 70):start]
    direct_right = folded[end:min(len(text), end + 70)]
    if (re.search(r"(?:" + DIRECT_PROCEEDING_ROLE.pattern[2:-2] + r")[\s:,-]*(?:(?:ong|ba)\s*)?$", direct_left)
            or re.match(r"^\s*[,;:]?\s*(?:la\s+)?(?:" + DIRECT_PROCEEDING_ROLE.pattern[2:-2] + r")\b", direct_right)):
        return True

    head_end = header_zone_end(text)
    foot_start = signature_block_start(text)
    if start < head_end:
        left_limit, right_limit = 0, head_end
    elif foot_start is not None and start >= foot_start:
        left_limit, right_limit = foot_start, len(text)
    else:
        # Body mentions inherit a protected role only when the same linked
        # entity also has a mention in the formal opening or signature block.
        # A nearby procedural title in narrative text is not enough.
        return False
    # Signature titles and names are often aligned in separate columns, with
    # the distribution list between them. Keep a wider search inside that
    # already bounded footer section only.
    local_window = max(window, 500) if foot_start is not None and start >= foot_start else window
    left_start = max(left_limit, start - local_window)
    right_end = min(right_limit, end + window)
    left = text[left_start:start]
    right = text[end:right_end]
    folded_left = folded[left_start:start]
    folded_right = folded[end:right_end]
    return bool(
        COURT_ROLE_TITLES.search(left) or COURT_ROLE_TITLES.search(right)
        or FOLDED_COURT_ROLE_TITLES.search(folded_left)
        or FOLDED_COURT_ROLE_TITLES.search(folded_right)
    )


def is_proceeding_agency_context(text: str, start: int, end: int) -> bool:
    """True when an ORG span names a court, procuracy, or investigative body."""
    if not 0 <= start < end <= len(text):
        return False
    return bool(PROCEEDING_AGENCY.search(_fold_vietnamese(text[start:end])))


def classify_person_role(text: str, start: int, end: int) -> str:
    """Return ``"court_personnel"`` or ``"party"`` for a PER/ORG span."""
    return "court_personnel" if is_court_personnel_context(text, start, end) else "party"


def preserve_court_personnel_clusters(text: str, clusters: list[dict]) -> list[dict]:
    """Mark linked people/agencies in procedural roles as non-redactable.

    Protect direct mentions of named proceeding agencies and people identified
    by a procedural title. The formal composition and signature layout also
    identifies officials when the name and role are on nearby separate lines.
    Once one mention triggers, the decision is propagated to its whole linked
    cluster so later mentions receive the same treatment.
    """
    for cluster in clusters:
        mentions = cluster.get("mentions") or []
        protected = []
        for mention in mentions:
            label = str(mention.get("label") or cluster.get("label") or "").upper()
            if label not in {"PER", "ORG"}:
                continue
            start, end = mention.get("start"), mention.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            is_proceeding_person = label == "PER" and is_court_personnel_context(text, start, end)
            is_proceeding_agency = label == "ORG" and is_proceeding_agency_context(text, start, end)
            if is_proceeding_person or is_proceeding_agency:
                protected.append(mention)

        if not protected:
            continue

        cluster["procedural_status"] = "court_personnel_or_agency"
        cluster["anonymization_action"] = "preserve"
        cluster["anonymization_reason"] = "proceeding_person_or_agency_exception"
        protected_ids = {id(mention) for mention in protected}
        for mention in mentions:
            mention["procedural_status"] = (
                "court_personnel_or_agency"
                if id(mention) in protected_ids
                else "linked_to_court_personnel_or_agency"
            )
            mention["anonymization_action"] = "preserve"
            mention["anonymization_reason"] = "linked_entity_is_proceeding_person_or_agency"

    return clusters
