"""Deterministic quality scoring for reconstructed legal documents.

The score is a triage signal, not a calibrated probability.  It is deliberately
based on pipeline evidence and validation results, never on a model's claimed
confidence alone.
"""

from collections import Counter


QUALITY_VERSION = "quality-v1"

# These findings mean that at least one identity or replacement decision is not
# safe to use as automatic training data.  They remain in the audit stream for
# later review instead of being silently discarded by the reconstruction step.
REASON_PENALTIES = {
    "missing_ner_predictions": 35,
    "ner_inference_error": 50,
    "ner_source_length_mismatch": 35,
    "roundtrip_failed": 100,
    "ambiguous_identity": 28,
    "missing_identity_anchor": 25,
    "conflicting_name_prefixes": 25,
    "conflicting_location_parents": 25,
    "possible_unmasked_alias": 25,
    "unsupported_name_prefix": 25,
    "unsupported_foreign_address": 25,
    "llm_decision_requires_review": 22,
    "unconfirmed_marker": 20,
    "organization_without_form": 16,
    "province_alias_requires_review": 14,
    "location_without_address_structure": 12,
    "fragmented_initial": 12,
    "fragmented_surname": 12,
    "numeral_or_alias": 12,
    "road_number_or_alias": 12,
    "overlapping_replacement_spans": 100,
    "incompatible_name_initial": 20,
}

HARD_FAIL_REASONS = {
    "error",
    "missing_ner_predictions",
    "ner_inference_error",
    "ner_source_length_mismatch",
    "roundtrip_failed",
    "ambiguous_identity",
    "missing_identity_anchor",
    "conflicting_name_prefixes",
    "conflicting_location_parents",
    "possible_unmasked_alias",
    "unsupported_name_prefix",
    "unsupported_foreign_address",
    "llm_decision_requires_review",
    "unconfirmed_marker",
    "overlapping_replacement_spans",
    "incompatible_name_initial",
}


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def collect_quality_reasons(audit, row=None):
    """Return unique reason counts from a current or legacy audit record."""
    audit = _as_dict(audit)
    row = _as_dict(row)
    counts = Counter()

    if audit.get("error"):
        counts["error"] += 1
    if row.get("error"):
        counts["error"] += 1
    for reason in audit.get("review_reasons", []) or []:
        counts[str(reason)] += 1
    for entity in audit.get("entities", []) or []:
        for reason in entity.get("review_reasons", []) or []:
            counts[str(reason)] += 1
    for mention in audit.get("mentions", []) or []:
        for reason in mention.get("review_reasons", []) or []:
            counts[str(reason)] += 1

    stats = _as_dict(audit.get("audit_stats"))
    if not stats:
        stats = _as_dict(row.get("reconstruction_stats"))
    if stats.get("roundtrip_verified") is False:
        counts["roundtrip_failed"] += 1
    if stats.get("overlap_conflict_entities", 0):
        counts["overlapping_replacement_spans"] += int(stats["overlap_conflict_entities"])

    return counts


def score_document(audit, row=None, min_replacements=1):
    """Score one document and return an auditable quality decision.

    Repeated instances of a reason are capped per reason so a long judgment
    with one systematic issue is not penalized merely for being long.
    """
    audit = _as_dict(audit)
    row = _as_dict(row)
    reasons = collect_quality_reasons(audit, row)
    stats = _as_dict(audit.get("audit_stats")) or _as_dict(row.get("reconstruction_stats"))
    replacements = int(stats.get("replacements", len(audit.get("replacements", []) or [])) or 0)
    roundtrip = stats.get("roundtrip_verified")
    if roundtrip is None:
        roundtrip = True

    penalty = 0
    applied_penalties = {}
    for reason, count in sorted(reasons.items()):
        unit = REASON_PENALTIES.get(reason, 10)
        applied = min(unit * min(count, 3), unit * 2)
        applied_penalties[reason] = applied
        penalty += applied

    if replacements < min_replacements:
        reasons["insufficient_replacements"] += 1
        applied_penalties["insufficient_replacements"] = 20
        penalty += 20

    score = max(0, min(100, 100 - penalty))
    hard_fail_reasons = sorted(reason for reason in reasons if reason in HARD_FAIL_REASONS)
    eligible = bool(roundtrip) and not hard_fail_reasons and replacements >= min_replacements
    if eligible and score >= 85:
        tier = "clean"
    elif score >= 60:
        tier = "review"
    else:
        tier = "reject"

    return {
        "quality_version": QUALITY_VERSION,
        "quality_score": score,
        "quality_tier": tier,
        "eligible": eligible,
        "roundtrip_verified": bool(roundtrip),
        "replacements": replacements,
        "min_replacements": min_replacements,
        "reasons": sorted(reasons),
        "reason_counts": dict(sorted(reasons.items())),
        "hard_fail_reasons": hard_fail_reasons,
        "penalties": applied_penalties,
    }
