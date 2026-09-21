"""Heuristic split of person/organisation mentions into court personnel versus
case participants.

Vietnamese court judgments are highly templated: a fixed set of role titles
introduces the presiding judge, assessors and clerk near the top of the
document and again in the closing signature block, while parties are only
ever referenced through declared roles (``nguyên đơn``, ``bị đơn``, ...) or
masked markers in the body. That structure is what a human reader relies on
to tell "the judge who signed this" from "the litigant it is about" -- no
statistics or training data are needed to reproduce it.

This module answers one narrow question for a future production anonymiser
that has to decide what to redact from an *un-anonymised* document: is a
given person/organisation mention sitting in a court-role heading or the
signature block? It is not used by the reconstruction pipeline in
``core.reconstruction``, which only ever fills in already-masked markers and
never makes a fresh masking decision on unmasked text.
"""
from __future__ import annotations

import re

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

# The closing block of a judgment: everything from here to the end of the
# document is signatures and the distribution list ("Nơi nhận"), never a
# party reference. Unlike COURT_ROLE_TITLES, these two phrases are
# closing-only conventions -- "TM." abbreviates "thay mặt" (on behalf of)
# and only introduces a signature, and "Nơi nhận" only introduces the
# distribution list. Neither appears in the opening composition list, which
# is why they -- and not COURT_ROLE_TITLES -- are safe to use as a hard
# cutoff for "everything after this point is the footer".
FOOTER_START = re.compile(r"(TM\.\s*HỘI ĐỒNG XÉT XỬ|Nơi nhận\s*:)", re.IGNORECASE)

# The opening composition list ("Thẩm phán - Chủ tọa phiên tòa: Ông ...")
# always ends before the first party is formally declared. This mirrors
# ``DECLARATION`` in ``core.reconstruction`` -- the same anchor the
# reconstruction engine already trusts to mean "a party is being named here".
FIRST_PARTY_DECLARATION = re.compile(
    r"(nguyên đơn|bị đơn|bị cáo|bị hại|người làm chứng|người khởi kiện|người bị kiện)\s*:",
    re.IGNORECASE,
)


def signature_block_start(text: str) -> int | None:
    """Offset where the judgment's closing signature block begins, if present.

    Uses the LAST match: a document can legitimately repeat court-role
    language earlier (see COURT_ROLE_TITLES), and the footer is always the
    final such occurrence, never the first.
    """
    matches = list(FOOTER_START.finditer(text))
    return matches[-1].start() if matches else None


def header_zone_end(text: str) -> int:
    """Offset where the opening court-composition list ends, or 0 if there is
    no declared party to bound it against (safe fallback: no header zone)."""
    match = FIRST_PARTY_DECLARATION.search(text)
    return match.start() if match else 0


def is_court_personnel_context(text: str, start: int, end: int, window: int = 120) -> bool:
    """True when the mention at ``text[start:end]`` sits in the opening
    court-composition list or the closing signature block.

    The local ``window`` check only looks inside the body region strictly
    between those two zones. Without that clipping, a title text sitting
    just across either boundary could "bleed" onto a nearby body mention
    that has nothing to do with it -- the same "no document-wide
    propagation" principle ``core.reconstruction`` already follows for
    markers.
    """
    head_end = header_zone_end(text)
    if start < head_end:
        return True
    foot_start = signature_block_start(text)
    if foot_start is not None and start >= foot_start:
        return True
    right_limit = foot_start if foot_start is not None else len(text)
    left = text[max(head_end, start - window):start]
    right = text[end:min(right_limit, end + window)]
    return bool(COURT_ROLE_TITLES.search(left) or COURT_ROLE_TITLES.search(right))


def classify_person_role(text: str, start: int, end: int) -> str:
    """Return ``"court_personnel"`` or ``"party"`` for a PER/ORG span."""
    return "court_personnel" if is_court_personnel_context(text, start, end) else "party"
