"""Create deterministic, document-disjoint train/validation/test source splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def split_documents(input_file: Path, output_dir: Path, *, text_field: str = "markdown",
                    id_field: str = "case_id", validation_ratio: float = 0.1,
                    test_ratio: float = 0.1, seed: str = "vdt-source-split-v1") -> dict:
    if (not math.isfinite(validation_ratio) or not math.isfinite(test_ratio)
            or validation_ratio < 0 or test_ratio < 0
            or validation_ratio + test_ratio >= 1):
        raise ValueError("split ratios must be nonnegative and sum to less than 1")
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = output_dir / "documents"
    assignment_path = output_dir / "split_assignments.jsonl"
    manifest_path = output_dir / "manifest.json"
    if documents_dir.exists() or assignment_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Split outputs already exist under {output_dir}; choose a new directory")
    documents_dir.mkdir()

    counts = Counter()
    seen_ids: set[str] = set()
    handles = {
        name: (documents_dir / f"{name}.jsonl").open("x", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    try:
        with input_file.open("r", encoding="utf-8-sig") as source, assignment_path.open(
            "x", encoding="utf-8"
        ) as assignments:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if id_field == "auto":
                    doc_id = next((str(row[key]).strip() for key in
                                   ("case_id", "doc_name", "official_document_id", "id")
                                   if row.get(key) is not None and str(row[key]).strip()), "")
                else:
                    doc_id = str(row.get(id_field) or "").strip()
                text = row.get(text_field)
                if not doc_id:
                    raise ValueError(f"Source row {line_number} has no {id_field!r}")
                if not isinstance(text, str) or not text.strip():
                    counts["empty_text_skipped"] += 1
                    continue
                if doc_id in seen_ids:
                    raise ValueError(f"Duplicate source document ID {doc_id!r} at row {line_number}")
                seen_ids.add(doc_id)
                rank = int.from_bytes(
                    hashlib.sha256(f"{seed}\x1f{doc_id}".encode("utf-8")).digest()[:8],
                    "big",
                ) / 2**64
                if rank < test_ratio:
                    split = "test"
                elif rank < test_ratio + validation_ratio:
                    split = "validation"
                else:
                    split = "train"
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                assignments.write(json.dumps({"doc_id": doc_id, "split": split}, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    if not counts["train"] or not counts["validation"] or not counts["test"]:
        raise ValueError("The source collection must populate all three splits; increase its size or adjust ratios")
    manifest = {
        "format": "vdt-document-splits-v1",
        "source_file": input_file.name,
        "source_sha256": _digest(input_file),
        "text_field": text_field,
        "id_field": id_field,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "assignment_method": "stable SHA-256 bucket of seed and document ID",
        "document_disjoint": True,
        "assignment_file": assignment_path.name,
        "counts": dict(counts),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-field", default="markdown")
    parser.add_argument("--id-field", default="auto",
                        help="source ID field; auto prefers case_id, then doc_name, official_document_id, or id")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="vdt-source-split-v1")
    args = parser.parse_args()
    if (not math.isfinite(args.validation_ratio) or not math.isfinite(args.test_ratio)
            or not 0 <= args.validation_ratio < 1 or not 0 <= args.test_ratio < 1):
        parser.error("split ratios must be in [0, 1)")
    if args.validation_ratio + args.test_ratio >= 1:
        parser.error("validation and test ratios must sum to less than 1")
    result = split_documents(
        args.input_file.resolve(), args.output_dir.resolve(), text_field=args.text_field,
        id_field=args.id_field, validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio, seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
