"""Local-only FastAPI service for NER, document linking, and legal anonymization."""

from __future__ import annotations

import argparse
import hmac
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import zipfile
from io import BytesIO

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..runtime import AnonymizationRuntime, Settings


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class TextRequest(BaseModel):
    text: str
    doc_id: str | None = None


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
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
