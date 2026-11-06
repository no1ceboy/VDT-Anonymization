"""Run the complete anonymization pipeline on DOCX, text, JSON, or JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from io import BytesIO
import zipfile

from .runtime import AnonymizationRuntime, Settings


def _settings(args: argparse.Namespace) -> Settings:
    ner_model = args.ner_model.expanduser().resolve()
    linker_base = args.linker_base_model.expanduser().resolve()
    linker_checkpoint = args.linker_checkpoint.expanduser().resolve()
    ocr_model = args.ocr_ngram_model.expanduser().resolve() if args.ocr_ngram_model else None

    if not ner_model.is_dir():
        raise ValueError(f"NER model directory does not exist: {ner_model}")
    if not linker_base.is_dir():
        raise ValueError(f"Linker base-model directory does not exist: {linker_base}")
    if not linker_checkpoint.is_file():
        raise ValueError(f"Linker checkpoint does not exist: {linker_checkpoint}")
    tokenizer_dir = linker_checkpoint.parent / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise ValueError(f"Linker tokenizer directory does not exist: {tokenizer_dir}")
    if ocr_model is not None and not ocr_model.is_file():
        raise ValueError(f"OCR spacing model does not exist: {ocr_model}")

    return Settings(
        ner_model_path=ner_model,
        linker_checkpoint=linker_checkpoint,
        linker_base_model=linker_base,
        ocr_ngram_model=ocr_model,
        api_key="",  # Local execution does not use API authentication.
        device=args.device,
        ner_dtype=args.ner_dtype,
        linker_amp_dtype=args.amp_dtype,
        cluster_threshold=args.threshold,
        ner_max_length=args.ner_max_length,
        ner_stride=args.ner_stride,
        ner_batch_size=args.ner_batch_size,
        min_ner_confidence=args.min_ner_confidence,
        max_chars=args.max_chars,
        max_mentions=args.max_mentions,
        max_docx_bytes=args.max_input_bytes,
        max_docx_expanded_bytes=args.max_expanded_bytes,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Detect, link, and anonymize entities in DOCX, UTF-8 text, "
                     "JSON, or JSONL documents.")
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input .docx, .txt, .json, or .jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Anonymized output file")
    parser.add_argument("--text-field", help="JSON/JSONL text field; auto-detects text, content, or markdown")
    parser.add_argument("--ner-model", type=Path, required=True, help="Local NER model directory")
    parser.add_argument("--linker-base-model", type=Path, required=True,
                        help="Local ELECTRA base-model directory")
    parser.add_argument("--linker-checkpoint", type=Path, required=True,
                        help="Trained pairwise linker checkpoint (best_model.pt)")
    parser.add_argument("--ocr-ngram-model", type=Path,
                        help="Optional local OCR spacing model JSON")
    parser.add_argument("--report", type=Path,
                        help="JSON report path (default: <output>.report.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing output and report files")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Document clustering threshold; default is the current calibrated 0.10")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--ner-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--ner-max-length", type=int, default=512)
    parser.add_argument("--ner-stride", type=int, default=128)
    parser.add_argument("--ner-batch-size", type=int, default=16)
    parser.add_argument("--min-ner-confidence", type=float, default=0.0)
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--max-mentions", type=int, default=1000)
    parser.add_argument("--max-input-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-expanded-bytes", type=int, default=200 * 1024 * 1024)
    args = parser.parse_args(argv)

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not 0 <= args.min_ner_confidence <= 1:
        parser.error("--min-ner-confidence must be between 0 and 1")
    if args.ner_max_length < 1 or args.ner_stride < 0:
        parser.error("--ner-max-length must be positive and --ner-stride cannot be negative")
    if args.ner_stride >= args.ner_max_length:
        parser.error("--ner-stride must be smaller than --ner-max-length")
    for name in (
        "ner_batch_size", "max_chars", "max_mentions", "max_input_bytes",
        "max_expanded_bytes",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    supported_suffixes = {".docx", ".txt", ".json", ".jsonl"}
    if args.input.suffix.casefold() not in supported_suffixes:
        parser.error("input must be a .docx, .txt, .json, or .jsonl file")
    if args.output.suffix.casefold() != args.input.suffix.casefold():
        parser.error("output extension must match the input (.docx, .txt, .json, or .jsonl)")
    if args.text_field and args.input.suffix.casefold() not in {".json", ".jsonl"}:
        parser.error("--text-field applies only to .json and .jsonl inputs")
    if args.output == args.input:
        parser.error("output must not overwrite the input")
    args.report = (
        args.report.expanduser().resolve()
        if args.report else args.output.with_suffix(args.output.suffix + ".report.json")
    )
    if args.report in {args.input, args.output}:
        parser.error("report path must differ from the input and output paths")
    if not args.overwrite and (args.output.exists() or args.report.exists()):
        parser.error("output or report already exists; choose new paths or pass --overwrite")
    return args


def _read_input(args: argparse.Namespace, settings: Settings):
    if args.input.stat().st_size > settings.max_docx_bytes:
        raise ValueError("Input file exceeds the configured size limit")
    if args.input.suffix.casefold() == ".txt":
        text = args.input.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError("Input text file is empty")
        return text, None, None

    from .api.docx_io import extract_docx_text, load_docx

    payload = args.input.read_bytes()
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            entries = archive.infolist()
            expanded = sum(item.file_size for item in entries)
            names = archive.namelist()
    except zipfile.BadZipFile as error:
        raise ValueError("Input is not a valid DOCX archive") from error
    if len(entries) > 5000 or expanded > settings.max_docx_expanded_bytes:
        raise ValueError("DOCX archive exceeds safety limits")
    if "word/document.xml" not in names:
        raise ValueError("Input is not a DOCX document")
    document = load_docx(payload)
    text, parts = extract_docx_text(document)
    return text, document, parts


def _read_structured_input(args: argparse.Namespace) -> tuple[object, list[object]]:
    """Read one JSON document, a JSON array, or JSONL document records."""
    if args.input.stat().st_size > args.max_input_bytes:
        raise ValueError("Input file exceeds the configured size limit")
    raw = args.input.read_text(encoding="utf-8-sig")
    if args.input.suffix.casefold() == ".json":
        payload = json.loads(raw)
        rows = payload if isinstance(payload, list) else [payload]
    else:
        payload = None
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("JSON input contains no documents")
    return payload, rows


def _record_text(record: object, requested_field: str | None,
                 row_number: int) -> tuple[str, str | None, str]:
    if isinstance(record, str):
        return record, None, f"document_{row_number:06d}"
    if not isinstance(record, dict):
        raise ValueError(
            f"JSON record {row_number} must be a string or an object containing a text field"
        )
    if requested_field:
        field = requested_field
        if field not in record:
            raise ValueError(f"JSON record {row_number} has no --text-field '{field}'")
    else:
        field = next((name for name in ("text", "content", "markdown") if name in record), None)
        if field is None:
            raise ValueError(
                f"JSON record {row_number} has no text/content/markdown field; pass --text-field"
            )
    text = record[field]
    if not isinstance(text, str):
        raise ValueError(f"JSON record {row_number} field '{field}' must contain a string")
    doc_id = next(
        (str(record[key]) for key in ("case_id", "doc_id", "document_id", "id")
         if record.get(key) is not None and str(record[key]).strip()),
        f"document_{row_number:06d}",
    )
    return text, field, doc_id


def _anonymize_records(rows: list[object], runtime: AnonymizationRuntime,
                       requested_field: str | None) -> tuple[list[object], list[dict]]:
    outputs: list[object] = []
    document_reports: list[dict] = []
    for row_number, record in enumerate(rows, start=1):
        text, field, doc_id = _record_text(record, requested_field, row_number)
        result = runtime.predict(text, doc_id)
        document_reports.append({
            "document_id": doc_id,
            "text_field": field,
            "status": result["status"],
            "review_required": result["review_required"],
            "ner_mentions": result["ner_mentions"],
            "clusters": len(result.get("clusters", [])),
            "threshold": result.get("threshold"),
            "ocr_normalization": result.get("ocr_normalization"),
            "anonymization_plan": result.get("anonymization_plan"),
            "mentions": result.get("mentions", []),
        })
        if result["status"] != "ready":
            outputs.append(None)
        elif field is None:
            outputs.append(result["anonymized_text"])
        else:
            anonymized_record = dict(record)
            anonymized_record[field] = result["anonymized_text"]
            outputs.append(anonymized_record)
    return outputs, document_reports


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        settings = _settings(args)
        structured_input = args.input.suffix.casefold() in {".json", ".jsonl"}
        if structured_input:
            payload, rows = _read_structured_input(args)
        else:
            text, document, parts = _read_input(args, settings)
        runtime = AnonymizationRuntime(settings)
        if structured_input:
            outputs, document_reports = _anonymize_records(rows, runtime, args.text_field)
            needs_review = any(item["review_required"] for item in document_reports)
            report = {
                "input": str(args.input),
                "output": str(args.output) if not needs_review else None,
                "status": "review_required" if needs_review else "ready",
                "review_required": needs_review,
                "document_count": len(rows),
                "documents": document_reports,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if needs_review:
                print(
                    f"Review required for one or more records; no anonymized file was written. "
                    f"Report: {args.report}",
                    file=sys.stderr,
                )
                raise SystemExit(3)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.input.suffix.casefold() == ".jsonl":
                output_text = "\n".join(
                    json.dumps(item, ensure_ascii=False) for item in outputs
                ) + "\n"
            else:
                output_payload = outputs if isinstance(payload, list) else outputs[0]
                output_text = json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n"
            args.output.write_text(output_text, encoding="utf-8")
            print(f"Anonymized file: {args.output}")
            print(f"Review report:   {args.report}")
            print(f"Processed documents: {len(rows)}")
            return

        result = runtime.predict(text, args.input.stem)

        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input": str(args.input),
            "output": str(args.output) if result["status"] == "ready" else None,
            "status": result["status"],
            "review_required": result["review_required"],
            "ner_mentions": result["ner_mentions"],
            "clusters": len(result.get("clusters", [])),
            "threshold": result.get("threshold"),
            "ocr_normalization": result.get("ocr_normalization"),
            "anonymization_plan": result.get("anonymization_plan"),
            "mentions": result.get("mentions", []),
        }
        if result["status"] != "ready":
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"Review required; no anonymized document was written. Report: {args.report}",
                file=sys.stderr,
            )
            raise SystemExit(3)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if document is None:
            args.output.write_text(result["anonymized_text"], encoding="utf-8")
        else:
            from .api.docx_io import apply_docx_replacements

            output = apply_docx_replacements(
                document, text, parts, result["anonymization_plan"]["replacements"]
            )
            args.output.write_bytes(output)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Anonymized file: {args.output}")
        print(f"Review report:   {args.report}")
        print(f"NER mentions: {report['ner_mentions']}; linked entities: {report['clusters']}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
