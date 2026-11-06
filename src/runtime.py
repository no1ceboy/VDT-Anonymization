"""Shared local NER, entity-linking, and anonymization runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

from .entity_linking.inference import DocumentEntityLinker
from .pipeline.ocr_ngram import CharacterTrigramModel, normalize_before_ner


def _env_path(name: str, required: bool = True) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        if required:
            raise RuntimeError(f"Required environment variable is not set: {name}")
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"{name} path does not exist: {path}")
    return path


@dataclass(frozen=True)
class Settings:
    ner_model_path: Path
    linker_checkpoint: Path
    linker_base_model: Path
    ocr_ngram_model: Path | None
    api_key: str
    device: str
    ner_dtype: str
    linker_amp_dtype: str
    cluster_threshold: float
    ner_max_length: int
    ner_stride: int
    ner_batch_size: int
    min_ner_confidence: float
    max_chars: int
    max_mentions: int
    max_docx_bytes: int
    max_docx_expanded_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.environ.get("VDT_API_KEY", "")
        if len(api_key) < 24:
            raise RuntimeError("Set VDT_API_KEY to a secret of at least 24 characters")
        threshold = float(os.environ.get("VDT_CLUSTER_THRESHOLD", "0.10"))
        if not 0 <= threshold <= 1:
            raise RuntimeError("VDT_CLUSTER_THRESHOLD must be between 0 and 1")
        ner_model_path = _env_path("VDT_NER_MODEL_PATH")
        linker_checkpoint = _env_path("VDT_LINKER_CHECKPOINT")
        linker_base_model = _env_path("VDT_LINKER_BASE_MODEL")
        ocr_ngram_model = _env_path("VDT_OCR_NGRAM_MODEL", required=False)
        for name, path in (("VDT_NER_MODEL_PATH", ner_model_path),
                           ("VDT_LINKER_BASE_MODEL", linker_base_model)):
            if not path.is_dir():
                raise RuntimeError(f"{name} must point to a local model directory: {path}")
        if not linker_checkpoint.is_file():
            raise RuntimeError(f"VDT_LINKER_CHECKPOINT must point to a checkpoint file: {linker_checkpoint}")
        if ocr_ngram_model is not None and not ocr_ngram_model.is_file():
            raise RuntimeError(f"VDT_OCR_NGRAM_MODEL must point to a model file: {ocr_ngram_model}")
        return cls(
            ner_model_path=ner_model_path,
            linker_checkpoint=linker_checkpoint,
            linker_base_model=linker_base_model,
            ocr_ngram_model=ocr_ngram_model,
            api_key=api_key,
            device=os.environ.get("VDT_DEVICE", "auto"),
            ner_dtype=os.environ.get("VDT_NER_DTYPE", "fp32"),
            linker_amp_dtype=os.environ.get("VDT_LINKER_AMP_DTYPE", "fp16"),
            cluster_threshold=threshold,
            ner_max_length=int(os.environ.get("VDT_NER_MAX_LENGTH", "512")),
            ner_stride=int(os.environ.get("VDT_NER_STRIDE", "128")),
            ner_batch_size=int(os.environ.get("VDT_NER_BATCH_SIZE", "16")),
            min_ner_confidence=float(os.environ.get("VDT_NER_MIN_CONFIDENCE", "0.0")),
            max_chars=int(os.environ.get("VDT_MAX_CHARS", "1000000")),
            max_mentions=int(os.environ.get("VDT_MAX_MENTIONS", "1000")),
            max_docx_bytes=int(os.environ.get("VDT_MAX_DOCX_BYTES", str(20 * 1024 * 1024))),
            max_docx_expanded_bytes=int(
                os.environ.get("VDT_MAX_DOCX_EXPANDED_BYTES", str(200 * 1024 * 1024))
            ),
        )


class AnonymizationRuntime:
    """Load models once and process documents without starting a service."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._inference_lock = Lock()

        from .pipeline.ner import (
            choose_device,
            infer_document,
            load_model,
            model_label_map,
            parse_entity_types,
        )

        self._infer_document = infer_document
        self._entity_types = parse_entity_types("PER,LOC,ORG")
        self.device = choose_device(settings.device)
        ner_args = SimpleNamespace(
            model_source="local",
            model_path=str(settings.ner_model_path),
            model_name=str(settings.ner_model_path),
            local_files_only=True,
            dtype=settings.ner_dtype,
        )
        self.ner_tokenizer, self.ner_model, _ = load_model(ner_args, self.device)
        self._label_map = model_label_map(self.ner_model)

        self.linker = DocumentEntityLinker(
            checkpoint_path=settings.linker_checkpoint,
            model_name=str(settings.linker_base_model),
            tokenizer_path=settings.linker_checkpoint.parent / "tokenizer",
            device=str(self.device),
            fp16=self.device.type == "cuda",
            amp_dtype=settings.linker_amp_dtype,
            context_chars=160,
        )
        self.ocr_model = (
            CharacterTrigramModel.load(settings.ocr_ngram_model)
            if settings.ocr_ngram_model else None
        )

    def predict(self, text: str, doc_id: str) -> dict:
        settings = self.settings
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Document text must be a non-empty string")
        if len(text) > settings.max_chars:
            raise ValueError(f"Document exceeds the {settings.max_chars}-character limit")

        with self._inference_lock:
            normalized = normalize_before_ner(text, self.ocr_model) if self.ocr_model else None
            ner_text = normalized.text if normalized else text
            mentions = self._infer_document(
                ner_text,
                self.ner_tokenizer,
                self.ner_model,
                self.device,
                self._label_map,
                self._entity_types,
                settings.ner_max_length,
                settings.ner_stride,
                settings.ner_batch_size,
                settings.min_ner_confidence,
            )
            source_mentions = []
            for mention in mentions:
                item = dict(mention)
                if normalized:
                    source_span = normalized.source_span(int(item["start"]), int(item["end"]))
                    if source_span is None:
                        raise ValueError("NER returned a span that cannot map to source text")
                    item["start"], item["end"] = source_span
                item["surface"] = text[int(item["start"]):int(item["end"])]
                source_mentions.append(item)

            if len(source_mentions) > settings.max_mentions:
                raise ValueError(
                    f"Document has {len(source_mentions)} NER mentions; limit is "
                    f"{settings.max_mentions}. Split the document or raise the configured limit."
                )
            result = self.linker.predict_document(
                doc_id, text, source_mentions, threshold=settings.cluster_threshold
            )
        if not source_mentions:
            result["anonymized_text"] = None
            result["anonymization_plan"]["status"] = "review_required"
            result["anonymization_plan"].setdefault("review_reasons", []).append(
                "no_supported_entity_mentions_detected"
            )
        result["status"] = result["anonymization_plan"]["status"]
        result["review_required"] = result["status"] != "ready"
        result["ner_mentions"] = len(source_mentions)
        result["ocr_normalization"] = (
            {
                "whitespace_changes": normalized.whitespace_changes,
                "joined_fragments": len(normalized.joins or []),
                "applied": True,
            }
            if normalized else {"applied": False}
        )
        return result
