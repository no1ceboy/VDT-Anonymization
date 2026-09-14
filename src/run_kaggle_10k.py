"""Stream court judgments and build a strict, diverse 10k reconstruction set.

This is the production entry point used by the Kaggle kernel.  It deliberately
keeps model inference, reconstruction, validation, and selection in one
process so the NER checkpoint is loaded only once.  Every durable output is
append-only or atomically replaced, making a graceful Kaggle restart safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .build_synthetic_unanonymized import process_row
    from .reconstruction import VERSION, TOKEN_RE
    from .run_ner import (
        DEFAULT_MODEL,
        choose_device,
        infer_document,
        load_model,
        model_label_map,
        parse_entity_types,
    )
    from .synthetic_lexicon import FAMILY_NAMES
except ImportError:
    from build_synthetic_unanonymized import process_row
    from reconstruction import VERSION, TOKEN_RE
    from run_ner import (
        DEFAULT_MODEL,
        choose_device,
        infer_document,
        load_model,
        model_label_map,
        parse_entity_types,
    )
    from synthetic_lexicon import FAMILY_NAMES


RUN_VERSION = "kaggle-clean-10k-v1"
DEFAULT_DATASET = "tmquan/cbba-toaan-gov-vn"
DEFAULT_CONFIG = "documents"

LETTER = r"[A-ZÀ-ỴĐ]"
WORD = rf"{LETTER}[^\W\d_]*"
MARKER = rf"(?:NLQ\s*\d+|NLC\s*\d+|(?:{LETTER}\.){{1,5}}{LETTER}(?:[1-9]\d*)?|(?:Th|Ph|Tr|Ng|Ch|Kh|Nh)\s*[1-9]\d*|{LETTER}\s*[1-9]\d*|{LETTER})"
FAMILY = "|".join(re.escape(name) for name in sorted(FAMILY_NAMES, key=len, reverse=True))

FULL_NAME_MARKER_RE = re.compile(
    rf"(?<!\w)(?:{FAMILY})\s+(?:{WORD}\s+){{0,4}}{MARKER}(?!\w)"
)
PERSON_CONTEXT_RE = re.compile(
    rf"(?<!\w)(?i:ông|bà|anh|chị|cháu|cô|chú|bác|em|nguyên đơn|bị đơn|bị cáo|"
    rf"bị hại|người làm chứng|người khởi kiện|người bị kiện)\s*:?\s*(?:{WORD}\s+){{0,4}}{MARKER}(?!\w)",
)
ADDRESS_RE = re.compile(
    rf"(?<!\w)(?i:số(?: nhà)?|tầng|lầu|phòng|căn|thửa|lô|ấp|xóm|thôn|làng|khóm|"
    rf"khu phố|tổ(?: dân phố)?|xã|phường|thị trấn|huyện|quận|thị xã|thành phố|tỉnh|"
    rf"đường|lộ|quốc lộ|ngõ|hẻm)\s+{MARKER}(?!\w)",
)
ORG_RE = re.compile(
    rf"(?<!\w)(?i:ngân hàng(?: thương mại cổ phần)?|công ty(?: trách nhiệm hữu hạn| cổ phần| tnhh)?|"
    rf"hợp tác xã|chi cục(?: thi hành án dân sự)?|ủy ban nhân dân)\s+{MARKER}(?!\w)",
)
PROCEDURAL_RE = re.compile(r"(?<!\w)(?:NLQ|NLC)\s*\d+(?!\w)", re.I)
DOTTED_RE = re.compile(rf"(?<!\w)(?:{LETTER}\.){{1,5}}{LETTER}(?:[1-9]\d*)?(?!\w)")
NUMBERED_MULTI_RE = re.compile(r"(?<!\w)(?:Th|Ph|Tr|Ng|Ch|Kh|Nh)\s*[1-9]\d*(?!\w)")
NUMBERED_RE = re.compile(rf"(?<!\w){LETTER}\s*[1-9]\d*(?!\w)")
ALIAS_RE = re.compile(r"\b(?:tên gọi khác|còn gọi là|bí danh|tên khác)\b", re.I)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(handle, record) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def document_id(row, source_number: int) -> str:
    for field in ("doc_name", "case_id", "official_document_id", "id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"source_{source_number}"


def category_of(row) -> str:
    value = row.get("category") or row.get("case_type") or "Unknown"
    return str(value).strip() or "Unknown"


def instance_of(row) -> str:
    value = row.get("instance_level") or row.get("cap_xet_xu") or "Unknown"
    return str(value).strip() or "Unknown"


def feature_profile(text: str) -> dict:
    """Fast pre-NER profile used to avoid inference on irrelevant documents."""
    counts = {
        "full_name_marker": len(FULL_NAME_MARKER_RE.findall(text)),
        "person_context": len(PERSON_CONTEXT_RE.findall(text)),
        "address_marker": len(ADDRESS_RE.findall(text)),
        "organization_marker": len(ORG_RE.findall(text)),
        "procedural_code": len(PROCEDURAL_RE.findall(text)),
        "dotted_initial": len(DOTTED_RE.findall(text)),
        "numbered_multi": len(NUMBERED_MULTI_RE.findall(text)),
        "numbered_marker": len(NUMBERED_RE.findall(text)),
        "alias_language": len(ALIAS_RE.findall(text)),
    }
    # TOKEN_RE is intentionally broader than the context patterns.  It is
    # recorded for diagnostics but cannot by itself admit a document.
    counts["all_marker_tokens"] = sum(1 for _ in TOKEN_RE.finditer(text))
    counts["relevant"] = int(any(counts[key] for key in (
        "full_name_marker", "person_context", "address_marker",
        "organization_marker", "procedural_code", "dotted_initial",
        "numbered_multi", "numbered_marker",
    )))
    return counts


def primary_challenge(features: dict) -> str:
    for name in (
        "alias_language", "numbered_multi", "dotted_initial", "procedural_code",
        "address_marker", "organization_marker", "numbered_marker",
        "full_name_marker", "person_context",
    ):
        if features.get(name):
            return name
    return "simple"


def stable_rank(seed: str, doc_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest()[:8], "big")


class ChallengePool:
    """Small deterministic sample of rejected cases, balanced by failure reason."""

    def __init__(self, per_reason: int, seed: str):
        self.per_reason = per_reason
        self.seed = seed
        self.by_reason = defaultdict(list)

    def consider(self, doc_id: str, output_row: dict, audit: dict, features: dict) -> None:
        quality = output_row.get("reconstruction_quality") or {}
        reasons = quality.get("reasons") or ["unknown_rejection"]
        if not audit.get("replacements") or "ner_inference_error" in reasons:
            return
        record = {
            "doc_id": doc_id,
            "source_markdown": output_row.get("original_anonymized_markdown", ""),
            "synthetic_markdown": output_row.get("synthetic_markdown", ""),
            "category": category_of(output_row),
            "instance_level": instance_of(output_row),
            "features": features,
            "quality": quality,
            "review_reasons": audit.get("review_reasons", []),
            "reconstruction_stats": output_row.get("reconstruction_stats", {}),
        }
        rank = stable_rank(self.seed, doc_id)
        for reason in reasons:
            bucket = self.by_reason[str(reason)]
            if any(item[1]["doc_id"] == doc_id for item in bucket):
                continue
            bucket.append((rank, record))
            bucket.sort(key=lambda item: item[0])
            del bucket[self.per_reason:]

    def restore(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_id = str(record.get("doc_id", ""))
                reasons = record.get("sampled_for_reasons") or (record.get("quality") or {}).get("reasons") or []
                rank = stable_rank(self.seed, doc_id)
                for reason in reasons:
                    bucket = self.by_reason[str(reason)]
                    bucket.append((rank, record))
                    bucket.sort(key=lambda item: item[0])
                    del bucket[self.per_reason:]

    def records(self):
        chosen = {}
        for reason, bucket in sorted(self.by_reason.items()):
            for _, record in bucket:
                item = chosen.setdefault(record["doc_id"], dict(record, sampled_for_reasons=[]))
                item["sampled_for_reasons"].append(reason)
        return [chosen[key] for key in sorted(chosen)]


def compact_clean_row(output_row: dict, source_number: int, features: dict) -> dict:
    record = dict(output_row)
    # process_row preserves `markdown` and also writes the explicitly named
    # source field. Keep only one source copy in a potentially large dataset.
    record.pop("markdown", None)
    record["curation"] = {
        "run_version": RUN_VERSION,
        "source_row_number": source_number,
        "strict_policy": "score_100_no_reasons_roundtrip_with_replacement",
        "challenge_features": features,
        "primary_challenge": primary_challenge(features),
    }
    return record


def compact_map(doc_id: str, audit: dict) -> dict:
    return {
        "doc_id": doc_id,
        "pipeline_version": audit.get("pipeline_version", VERSION),
        "source_sha256": audit.get("source_sha256"),
        "entities": [
            {
                "entity_id": entity.get("entity_id"),
                "label": entity.get("label"),
                "marker": entity.get("marker"),
                "role": entity.get("role"),
                "person_anchor": entity.get("person_anchor"),
                "synthetic_value": entity.get("synthetic_value"),
                "reconstructable": entity.get("reconstructable"),
                "name_rule": entity.get("name_rule"),
            }
            for entity in audit.get("entities", [])
            if entity.get("reconstructable")
        ],
        "replacements": audit.get("replacements", []),
    }


def load_existing(path: Path):
    ids = set()
    categories = Counter()
    instances = Counter()
    challenges = Counter()
    if not path.exists():
        return ids, categories, instances, challenges
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.add(document_id(row, len(ids)))
            categories[category_of(row)] += 1
            instances[instance_of(row)] += 1
            challenges[(row.get("curation") or {}).get("primary_challenge", "simple")] += 1
    return ids, categories, instances, challenges


def selection_allowed(category: str, challenge: str, accepted: int, target: int,
                      category_counts: Counter, challenge_counts: Counter) -> bool:
    """Soft caps prevent one common case type or easy pattern dominating."""
    # Caps are deliberately soft near the end: once 90% is collected, filling
    # the exact target is more useful than stalling over a missing rare stratum.
    if accepted >= math.floor(target * 0.90):
        return True
    category_cap = max(50, math.ceil(target * 0.32))
    simple_cap = max(50, math.ceil(target * 0.38))
    if category_counts[category] >= category_cap:
        return False
    if challenge in {"person_context", "full_name_marker", "simple"} and challenge_counts[challenge] >= simple_cap:
        return False
    return True


def make_report(args, counters, category_counts, instance_counts, challenge_counts,
                started_at, complete, challenge_pool_size) -> dict:
    return {
        "run_version": RUN_VERSION,
        "pipeline_version": VERSION,
        "complete": complete,
        "target": args.target,
        "accepted": counters["accepted"],
        "source": {"dataset": args.dataset, "config": args.config, "split": args.split},
        "policy": {
            "quality_score": 100,
            "eligible": True,
            "reasons": [],
            "roundtrip_verified": True,
            "minimum_replacements": 1,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "category_soft_cap_fraction": 0.32,
            "simple_feature_soft_cap_fraction": 0.38,
            "soft_caps_relaxed_after_fraction": 0.90,
            "llm_auto_acceptance": False,
        },
        "counts": dict(sorted(counters.items())),
        "category_distribution": dict(sorted(category_counts.items())),
        "instance_distribution": dict(sorted(instance_counts.items())),
        "challenge_distribution": dict(sorted(challenge_counts.items())),
        "challenge_pool_documents": challenge_pool_size,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "updated_at_epoch": time.time(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", default=DEFAULT_DATASET)
    result.add_argument("--config", default=DEFAULT_CONFIG)
    result.add_argument("--split", default="train")
    result.add_argument("--output-dir", default="/kaggle/working/vdt_clean_10k")
    result.add_argument("--target", type=int, default=10_000)
    result.add_argument("--min-chars", type=int, default=500)
    result.add_argument("--max-chars", type=int, default=50_000)
    result.add_argument("--max-source-rows", type=int, default=0)
    result.add_argument("--max-runtime-minutes", type=int, default=690)
    result.add_argument("--checkpoint-every", type=int, default=100)
    result.add_argument("--challenge-per-reason", type=int, default=20)
    result.add_argument("--seed", default="vdt-clean-10k-2026")
    result.add_argument("--model-name", default=DEFAULT_MODEL)
    result.add_argument("--model-source", choices=["huggingface", "local"], default="huggingface")
    result.add_argument("--model-path")
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    result.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="fp16")
    result.add_argument("--entity-types", default="PER,LOC,ORG")
    result.add_argument("--max-length", type=int, default=512)
    result.add_argument("--stride", type=int, default=128)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--min-confidence", type=float, default=0.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.target < 1 or args.min_chars < 0 or args.max_chars < args.min_chars:
        raise SystemExit("invalid target or character limits")
    if args.checkpoint_every < 1 or args.max_runtime_minutes < 1 or args.challenge_per_reason < 1:
        raise SystemExit("checkpoint/runtime/challenge limits must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean_10000.jsonl"
    maps_path = output_dir / "clean_10000_maps.jsonl"
    challenge_path = output_dir / "challenge_rejected.jsonl"
    progress_path = output_dir / "progress.json"
    report_path = output_dir / "manifest.json"

    accepted_ids, category_counts, instance_counts, challenge_counts = load_existing(clean_path)
    counters = Counter(accepted=len(accepted_ids))
    source_resume = 0
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous.get("run_version") == RUN_VERSION:
            source_resume = int(previous.get("source_seen", 0))
            counters.update({k: int(v) for k, v in (previous.get("counts") or {}).items() if k != "accepted"})
    if len(accepted_ids) >= args.target:
        print(f"[10K] Already complete: {len(accepted_ids)} clean documents")
        return 0

    started_at = time.time()
    deadline = started_at + args.max_runtime_minutes * 60
    challenge_pool = ChallengePool(args.challenge_per_reason, args.seed)
    challenge_pool.restore(challenge_path)

    device = choose_device(args.device)
    tokenizer, model, model_reference = load_model(args, device)
    labels = model_label_map(model)
    entity_types = parse_entity_types(args.entity_types)
    print(f"[10K] device={device}; model={model_reference}; resume_source={source_resume}; accepted={len(accepted_ids)}")

    from datasets import load_dataset
    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    if source_resume:
        dataset = dataset.skip(source_resume)

    clean_handle = clean_path.open("a", encoding="utf-8", buffering=1)
    maps_handle = maps_path.open("a", encoding="utf-8", buffering=1)
    last_checkpoint_work = counters["candidates"]

    def checkpoint(source_seen: int, complete: bool = False) -> None:
        clean_handle.flush()
        maps_handle.flush()
        os.fsync(clean_handle.fileno())
        os.fsync(maps_handle.fileno())
        challenge_records = challenge_pool.records()
        atomic_jsonl(challenge_path, challenge_records)
        counters["accepted"] = len(accepted_ids)
        progress = {
            "run_version": RUN_VERSION,
            "pipeline_version": VERSION,
            "source_seen": source_seen,
            "accepted": len(accepted_ids),
            "complete": complete,
            "counts": dict(counters),
            "category_distribution": dict(category_counts),
            "instance_distribution": dict(instance_counts),
            "challenge_distribution": dict(challenge_counts),
        }
        atomic_json(progress_path, progress)
        atomic_json(report_path, make_report(
            args, counters, category_counts, instance_counts, challenge_counts,
            started_at, complete, len(challenge_records),
        ))
        print(
            f"[10K] source={source_seen:,} candidates={counters['candidates']:,} "
            f"accepted={len(accepted_ids):,}/{args.target:,} rejected={counters['strict_rejected']:,}",
            flush=True,
        )

    source_seen = source_resume
    try:
        for relative_number, raw_row in enumerate(dataset):
            source_seen = source_resume + relative_number + 1
            counters["source_seen_this_run"] += 1
            if args.max_source_rows and source_seen >= args.max_source_rows:
                counters["source_limit_reached"] += 1
                break
            if time.time() >= deadline:
                counters["runtime_limit_reached"] += 1
                break

            row = dict(raw_row)
            text = row.get("markdown")
            if not isinstance(text, str) or len(text.strip()) < args.min_chars:
                counters["too_short_or_missing"] += 1
                continue
            if len(text) > args.max_chars:
                counters["too_long"] += 1
                continue
            doc_id = document_id(row, source_seen - 1)
            if doc_id in accepted_ids:
                counters["already_accepted"] += 1
                continue

            features = feature_profile(text)
            if not features["relevant"]:
                counters["not_marker_relevant"] += 1
                continue

            counters["candidates"] += 1
            try:
                entities = infer_document(
                    text, tokenizer, model, device, labels, entity_types,
                    args.max_length, args.stride, args.batch_size, args.min_confidence,
                )
                ner_record = {"doc_id": doc_id, "char_len": len(text), "entities": entities}
                _, output_row, audit, _ = process_row(source_seen - 1, row, ner_record, "markdown")
            except RuntimeError as exc:
                counters["inference_errors"] += 1
                print(f"[10K] inference error doc={doc_id}: {exc}", file=sys.stderr, flush=True)
                if getattr(device, "type", "") == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                continue
            except Exception as exc:
                counters["document_errors"] += 1
                print(f"[10K] document error doc={doc_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue

            quality = output_row.get("reconstruction_quality") or {}
            strict = (
                quality.get("eligible") is True
                and quality.get("quality_score") == 100
                and not quality.get("reasons")
                and quality.get("roundtrip_verified") is True
                and int(quality.get("replacements", 0)) >= 1
            )
            category = category_of(output_row)
            instance = instance_of(output_row)
            challenge = primary_challenge(features)
            if strict and selection_allowed(
                category, challenge, len(accepted_ids), args.target,
                category_counts, challenge_counts,
            ):
                clean_row = compact_clean_row(output_row, source_seen - 1, features)
                append_jsonl(clean_handle, clean_row)
                append_jsonl(maps_handle, compact_map(doc_id, audit))
                accepted_ids.add(doc_id)
                category_counts[category] += 1
                instance_counts[instance] += 1
                challenge_counts[challenge] += 1
                counters["accepted"] = len(accepted_ids)
            elif strict:
                counters["deferred_by_diversity_cap"] += 1
            else:
                counters["strict_rejected"] += 1
                challenge_pool.consider(doc_id, output_row, audit, features)

            if len(accepted_ids) >= args.target:
                checkpoint(source_seen, complete=True)
                return 0
            if counters["candidates"] - last_checkpoint_work >= args.checkpoint_every:
                checkpoint(source_seen)
                last_checkpoint_work = counters["candidates"]
    except KeyboardInterrupt:
        counters["interrupted"] += 1
        print("[10K] Interrupted; publishing the latest checkpoint for resume.", file=sys.stderr, flush=True)
    except Exception as exc:
        counters["stream_errors"] += 1
        print(f"[10K] Safe stop after source/model error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        checkpoint(source_seen, complete=len(accepted_ids) >= args.target)
        clean_handle.close()
        maps_handle.close()

    if len(accepted_ids) < args.target:
        print(
            f"[10K] Safe stop with {len(accepted_ids):,}/{args.target:,} accepted. "
            f"Rerun with the same output directory to resume at source row {source_seen:,}.",
            flush=True,
        )
        # A controlled incomplete run is successful from Kaggle's perspective,
        # ensuring its checkpoint files are published and can seed a resume.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
