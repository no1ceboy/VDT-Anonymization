"""CLI for the staged legal reconstruction pipeline. See reconstruction.py."""
import argparse
import json
import os
import hashlib
from collections import Counter

try:
    from .legal_linking import GeminiResolver, load_api_key, verify_roundtrip
    from .reconstruction import VERSION, link_document, apply_replacements, marker_value, observations
except ImportError:
    from legal_linking import GeminiResolver, load_api_key, verify_roundtrip
    from reconstruction import VERSION, link_document, apply_replacements, marker_value, observations

DEFAULT_SOURCE = "datasets/legal_test.jsonl"
DEFAULT_NER = "outputs/nlphust_legal_test.jsonl"
DEFAULT_DATASET_OUTPUT = "outputs/synthetic_unanonymized.jsonl"
DEFAULT_LINK_OUTPUT = "outputs/entity_links.jsonl"
DEFAULT_MAP_OUTPUT = "outputs/replacement_maps.jsonl"

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


def process_row(row_number, row, ner_record, text_field, resolver=None):
    doc_id = str(row.get("case_id", row.get("doc_name", row.get("id", row_number))))
    source_text = row.get(text_field)
    if source_text is None:
        return doc_id, dict(row), {"doc_id": doc_id, "error": f"missing field: {text_field}"}, None

    source_text = str(source_text)
    document_issues = []
    if ner_record is None:
        document_issues.append('missing_ner_predictions')
    elif ner_record.get('char_len') not in (None, len(source_text)):
        document_issues.append('ner_source_length_mismatch')
        ner_record = None
    mentions, entities, replacements = link_document(doc_id, source_text, ner_record, resolver)
    synthetic_text, applied = apply_replacements(source_text, replacements)
    verify_roundtrip(source_text, synthetic_text, applied)
    output_row = dict(row)
    output_row["synthetic_markdown"] = synthetic_text
    output_row["original_anonymized_markdown"] = source_text
    output_row["synthetic_reconstruction"] = True
    review_reasons = sorted(set(document_issues) | {reason for entity in entities for reason in entity.get('review_reasons', [])}
                            | {reason for mention in mentions for reason in mention.get('review_reasons', [])})
    review_status = 'needs_review' if review_reasons else 'passed_automatic_checks'
    output_row['pipeline_version'] = VERSION
    output_row['review_status'] = review_status
    output_row['review_reasons'] = review_reasons
    output_row["reconstruction_stats"] = {
        "roundtrip_verified": True,
        "review_entities": sum(bool(e.get('review_reasons')) for e in entities),
        "overlap_conflict_entities": sum('overlapping_replacement_spans' in e.get('review_reasons', []) for e in entities),
        "rejected_candidates": sum(m.get('link_status') == 'rejected_nonentity' for m in mentions),
        "unresolved_types": sum(e.get("link_status") == "unresolved_type" for e in entities),
        "llm_accepted_mentions": sum(m.get("link_status") == "linked_by_llm" for m in mentions),
        "skipped_overlapping_replacements": len(replacements) - len(applied),
        "mentions": len(mentions),
        "linked_entities": len(entities),
        "reconstructable_entities": sum(entity["reconstructable"] for entity in entities),
        "replacements": len(applied),
        "unmasked_evidence_entities": sum(
            entity.get("link_status") == "linked_to_unmasked_evidence" for entity in entities
        ),
        "ambiguous_markers": sum(
            entity.get("link_status") == "ambiguous_marker" for entity in entities
        ),
        "ocr_repaired_mentions": sum(
            bool(mention.get("ocr_prefix_marker_start") is not None)
            for mention in mentions
        ),
    }
    audit = {
        "row_number": row_number,
        "doc_id": doc_id,
        "pipeline_version": VERSION,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "review_status": review_status,
        "review_reasons": review_reasons,
        "rejected_candidates": [m for m in mentions if m.get('link_status') == 'rejected_nonentity'],
        "ner_observations": observations(source_text, ner_record),
        "audit_stats": output_row["reconstruction_stats"],
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
    parser.add_argument("--llm-provider", choices=["none", "gemini"], default="none")
    parser.add_argument("--llm-model", help="Gemini model ID; required with --llm-provider gemini")
    parser.add_argument("--env-file", help="Optional dotenv file containing GOOGLE_API_KEY or GEMINI_API_KEY")
    parser.add_argument("--llm-max-calls", type=int, default=20, help="Maximum paid requests for this run")
    parser.add_argument("--llm-timeout", type=int, default=45)
    args = parser.parse_args()
    resolver = None
    if args.llm_max_calls < 0 or args.llm_timeout <= 0:
        parser.error("LLM call budget must be nonnegative and timeout positive")
    if args.llm_provider == "gemini":
        if not args.llm_model:
            parser.error("--llm-model is required for Gemini")
        try:
            resolver = GeminiResolver(args.llm_model, load_api_key(args.env_file), args.llm_max_calls, args.llm_timeout)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    if not os.path.exists(args.input_file):
        parser.error(f"Input file does not exist: {args.input_file}")
    if args.limit < 0 or args.offset < 0:
        parser.error("--limit and --offset cannot be negative")

    ner_by_id = load_jsonl_by_id(args.ner_file)
    output_paths = [os.path.normcase(os.path.realpath(p)) for p in (args.output_file, args.links_file, args.maps_file)]
    input_paths = {os.path.normcase(os.path.realpath(p)) for p in (args.input_file, args.ner_file)}
    if len(set(output_paths)) != 3 or any(p in input_paths for p in output_paths):
        parser.error("The three output paths must be distinct and must not overwrite an input")
    for path in (args.output_file, args.links_file, args.maps_file):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    documents = 0
    errors = 0
    replacements = 0
    reconstructable = 0
    label_counts = Counter()
    llm_status_counts = Counter()

    with (
        open(args.output_file, "w", encoding="utf-8") as dataset_handle,
        open(args.links_file, "w", encoding="utf-8") as links_handle,
        open(args.maps_file, "w", encoding="utf-8") as maps_handle,
    ):
        for row_number, row in iter_source_rows(args.input_file, args.offset, args.limit):
            doc_id, output_row, audit, details = process_row(
                row_number, row, ner_by_id.get(str(row.get("case_id", row.get("doc_name", row.get("id", row_number))))), args.text_field, resolver
            )
            dataset_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            links_handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            map_record = {
                "row_number": row_number,
                "doc_id": doc_id,
                "pipeline_version": VERSION,
                "review_status": audit.get('review_status'),
                "entities": [
                    {
                        "entity_id": entity["entity_id"],
                        "label": entity["label"],
                        "marker": entity["marker"],
                        "person_anchor": entity.get("person_anchor"),
                        "encoding_scheme": entity.get("encoding_scheme"),
                        "procedural_role": entity.get("procedural_role"),
                        "known_full_name": entity.get("known_full_name"),
                        "synthetic_value": entity["synthetic_value"],
                        "reconstructable": entity["reconstructable"],
                        "review_reasons": entity.get('review_reasons', []),
                        "name_rule": entity["name_rule"],
                        "marker_initial_preserved": entity["marker_initial_preserved"],
                    }
                    for entity in audit.get("entities", [])
                ],
                "replacements": audit.get("replacements", []),
            }
            maps_handle.write(json.dumps(map_record, ensure_ascii=False) + "\n")

            documents += 1
            for entity in audit.get("entities", []):
                for mention in entity.get("mentions", []):
                    if mention.get("llm_decision"):
                        llm_status_counts[mention["llm_decision"]["status"]] += 1
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
    if resolver:
        print(f"[RECON] Gemini requests: {resolver.calls}/{resolver.max_calls}")
        print(f"[RECON] LLM audit statuses (per mention): {dict(llm_status_counts)}")
    print(f"[RECON] Dataset saved to: {args.output_file}")
    print(f"[RECON] Links saved to: {args.links_file}")
    print(f"[RECON] Maps saved to: {args.maps_file}")


if __name__ == "__main__":
    main()
