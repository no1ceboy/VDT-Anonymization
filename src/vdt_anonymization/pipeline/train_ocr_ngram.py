"""Train the local character-trigram model used by the OCR spacing pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .ocr_ngram import CharacterTrigramModel


def iter_texts(path: str, text_field: str, limit: int = 0):
    with open(path, "r", encoding="utf-8-sig") as handle:
        for row_number, line in enumerate(handle):
            if limit and row_number >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(text_field)
            if value is not None:
                yield str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Train a corpus-derived Vietnamese OCR spacing model"
    )
    parser.add_argument("--input-file", required=True, help="Legal corpus JSONL")
    parser.add_argument("--text-field", default="markdown", help="Document text field")
    parser.add_argument("--output-file", default="outputs/ocr_char_trigram.json",
                        help="Where to write the model JSON")
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of JSONL rows to read; 0 means all")
    parser.add_argument("--min-word-count", type=int, default=3,
                        help="Minimum corpus count for a candidate joined token")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error(f"Input file does not exist: {args.input_file}")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.min_word_count < 1:
        parser.error("--min-word-count must be positive")

    print(f"[OCR LM] Reading {args.input_file} ({args.text_field})")

    def report(documents, characters, tokens):
        print(f"[OCR LM] Documents={documents:,}; characters={characters:,}; tokens={tokens:,}")

    model = CharacterTrigramModel.train(
        iter_texts(args.input_file, args.text_field, args.limit),
        min_word_count=args.min_word_count,
        progress=report,
    )
    model.metadata["corpus_file"] = Path(args.input_file).name
    model.metadata["text_field"] = args.text_field
    model.save(args.output_file)
    size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print("[OCR LM] Training complete")
    print(f"[OCR LM] Documents: {model.metadata['documents']:,}")
    print(f"[OCR LM] Characters: {model.metadata['characters']:,}")
    print(f"[OCR LM] Candidate tokens: {model.metadata['candidate_words']:,}")
    print(f"[OCR LM] Model size: {size_mb:.1f} MiB")
    print(f"[OCR LM] Saved: {args.output_file}")


if __name__ == "__main__":
    main()
