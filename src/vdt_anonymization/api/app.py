"""Local-only FastAPI service for NER, document linking, and legal anonymization."""

from __future__ import annotations

import argparse
import hmac
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4
import zipfile
from io import BytesIO

import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..entity_linking.inference import DocumentEntityLinker
from ..pipeline.ocr_ngram import CharacterTrigramModel, normalize_before_ner


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


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
            max_mentions=int(os.environ.get("VDT_MAX_MENTIONS", "500")),
            max_docx_bytes=int(os.environ.get("VDT_MAX_DOCX_BYTES", str(20 * 1024 * 1024))),
            max_docx_expanded_bytes=int(
                os.environ.get("VDT_MAX_DOCX_EXPANDED_BYTES", str(200 * 1024 * 1024))
            ),
        )


class TextRequest(BaseModel):
    text: str
    doc_id: str | None = None


class AnonymizationRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._inference_lock = Lock()

        from ..pipeline.ner import (
            choose_device,
            infer_document,
            load_model,
            model_label_map,
            parse_entity_types,
        )

        self._infer_document = infer_document
        self._label_map = None
        self._entity_types = parse_entity_types("PER,LOC,ORG")
        self.device = choose_device(settings.device)
        ner_args = SimpleNamespace(
            model_source="local",
            model_path=str(settings.ner_model_path),
            model_name=str(settings.ner_model_path),
            local_files_only=True,
            dtype=settings.ner_dtype,
        )
        self.ner_tokenizer, self.ner_model, self.ner_model_reference = load_model(
            ner_args, self.device
        )
        self._label_map = model_label_map(self.ner_model)

        linker_fp16 = self.device.type == "cuda"
        self.linker = DocumentEntityLinker(
            checkpoint_path=settings.linker_checkpoint,
            model_name=str(settings.linker_base_model),
            tokenizer_path=settings.linker_checkpoint.parent / "tokenizer",
            device=str(self.device),
            fp16=linker_fp16,
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
                    f"{settings.max_mentions}. Split the document or raise VDT_MAX_MENTIONS."
                )
            result = self.linker.predict_document(
                doc_id, text, source_mentions, threshold=settings.cluster_threshold
            )
        if not source_mentions:
            # A clean-looking response with zero detected entities can conceal
            # a complete NER miss. Make the caller review it instead of
            # treating unchanged text as a successful anonymization.
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


def _require_api_key(request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.runtime = AnonymizationRuntime(settings)
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        app.state.runtime = None


app = FastAPI(
    title="VDT Vietnamese Court Anonymization API",
    version="1.0.0",
    description="Local NER, entity linking, clustering, and legal anonymization.",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz(request: Request):
    return {"status": "ready" if getattr(request.app.state, "ready", False) else "starting"}


@app.post("/v1/anonymize/text", dependencies=[Depends(_require_api_key)])
def anonymize_text(body: TextRequest, request: Request):
    runtime: AnonymizationRuntime = request.app.state.runtime
    doc_id = (body.doc_id or f"doc-{uuid4().hex[:12]}").strip()
    if not doc_id or len(doc_id) > 128:
        raise HTTPException(status_code=422, detail="doc_id must contain 1 to 128 characters")
    try:
        return runtime.predict(body.text, doc_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _check_docx_archive(payload: bytes, settings: Settings) -> None:
    if len(payload) > settings.max_docx_bytes:
        raise HTTPException(status_code=413, detail="DOCX upload exceeds the configured size limit")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            entries = archive.infolist()
            expanded = sum(item.file_size for item in entries)
            if len(entries) > 5000 or expanded > settings.max_docx_expanded_bytes:
                raise HTTPException(status_code=413, detail="DOCX archive exceeds safety limits")
            if "word/document.xml" not in archive.namelist():
                raise HTTPException(status_code=422, detail="Upload is not a DOCX document")
    except zipfile.BadZipFile as error:
        raise HTTPException(status_code=422, detail="Upload is not a valid DOCX archive") from error


@app.post("/v1/anonymize/docx", dependencies=[Depends(_require_api_key)])
async def anonymize_docx(
    request: Request,
    file: UploadFile = File(...),
):
    from .docx_io import apply_docx_replacements, extract_docx_text, load_docx

    settings: Settings = request.app.state.settings
    payload = await file.read(settings.max_docx_bytes + 1)
    _check_docx_archive(payload, settings)
    try:
        document = load_docx(payload)
        source_text, parts = extract_docx_text(document)
        doc_id = Path(file.filename or f"doc-{uuid4().hex[:12]}").stem
        result = request.app.state.runtime.predict(source_text, doc_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    if result["status"] != "ready":
        return JSONResponse(
            status_code=409,
            content={
                "doc_id": result["doc_id"],
                "status": result["status"],
                "review_required": True,
                "anonymization_plan": result["anonymization_plan"],
                "mentions": result["mentions"],
            },
        )

    try:
        output = apply_docx_replacements(
            document, source_text, parts, result["anonymization_plan"]["replacements"]
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    stem = Path(file.filename or "document").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80] or "document"
    filename = stem + "_anonymized.docx"
    return Response(
        content=output,
        media_type=DOCX_MIME,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-VDT-Status": "ready",
            "X-VDT-Doc-ID": result["doc_id"],
        },
    )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local VDT anonymization API")
    parser.add_argument("--host", default=os.environ.get("VDT_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VDT_API_PORT", "8000")))
    parser.add_argument("--log-level", choices=["critical", "error", "warning", "info"], default="info")
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in 1..65535")
    if int(os.environ.get("VDT_API_WORKERS", "1")) != 1:
        parser.error("The local model service requires one worker to avoid duplicate GPU/model memory")
    if not os.environ.get("VDT_API_KEY"):
        parser.error("Set VDT_API_KEY before starting the API")
    uvicorn.run(
        "vdt_anonymization.api.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
