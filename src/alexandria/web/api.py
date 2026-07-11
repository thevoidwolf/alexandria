"""JSON endpoints under /api/* for the browser UI (and curl scripting)."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from ulid import ULID

from alexandria.catalog import get_catalog, get_document, list_documents
from alexandria.config import Config
from alexandria.curate import (
    delete_category,
    delete_document,
    delete_tag,
    rename_category,
    rename_tag,
    update_document_metadata,
)
from alexandria.search import format_snippet_markdown, search
from alexandria.web.jobs import JobQueue

_VALID_MODES = {"hybrid", "fts", "vec"}


def _split_tags(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _version() -> str:
    try:
        return importlib_metadata.version("alexandria")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0+dev"


def _sanitize_filename(name: str) -> str:
    """Basename-only, non-empty, no leading dots."""
    name = (name or "").replace("\\", "/")
    name = Path(name).name.strip().lstrip(".")
    return name or "unnamed"


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_api_routes(
    cfg: Config,
    conn: sqlite3.Connection,
    lock: threading.Lock,
    jobs: JobQueue | None = None,
) -> list[Route]:
    async def api_info(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "version": _version(),
                "title": cfg.web.title,
                "home": str(cfg.home),
                "embed_model": cfg.embeddings.model,
                "embed_dim": cfg.embeddings.dim,
                "pdf_backend": cfg.extractors.pdf.backend,
                "max_upload_mb": cfg.web.max_upload_mb,
                "jobs_enabled": jobs is not None,
            }
        )

    async def api_catalog(_request: Request) -> JSONResponse:
        with lock:
            summary = get_catalog(conn)
        return JSONResponse(asdict(summary))

    async def api_search(request: Request) -> JSONResponse:
        params = request.query_params
        query = (params.get("q") or "").strip()
        if not query:
            return JSONResponse([])
        category = params.get("category") or None
        tags = _split_tags(params.get("tags") or "")
        try:
            limit = int(params.get("limit", "10"))
        except ValueError:
            return JSONResponse({"error": "invalid limit"}, status_code=400)
        if limit < 1 or limit > 100:
            return JSONResponse(
                {"error": "limit out of range (1..100)"}, status_code=400
            )
        mode = params.get("mode", "hybrid")
        if mode not in _VALID_MODES:
            return JSONResponse(
                {"error": f"invalid mode: {mode!r}"}, status_code=400
            )
        with lock:
            hits = search(
                query, conn, cfg,
                category=category, tags=tags, limit=limit, mode=mode,  # type: ignore[arg-type]
            )
        return JSONResponse(
            [
                {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "score": h.score,
                    "snippet": format_snippet_markdown(h.snippet),
                    "title": h.title,
                    "category": h.category,
                    "tags": h.tags,
                    "source_uri": h.source_uri,
                    "content_type": h.content_type,
                }
                for h in hits
            ]
        )

    async def api_list_documents(request: Request) -> JSONResponse:
        params = request.query_params
        category = params.get("category") or None
        tags = _split_tags(params.get("tags") or "")
        since = params.get("since") or None
        try:
            limit = int(params.get("limit", "50"))
            offset = int(params.get("offset", "0"))
        except ValueError:
            return JSONResponse({"error": "invalid limit/offset"}, status_code=400)
        if limit < 1 or limit > 500 or offset < 0:
            return JSONResponse(
                {"error": "limit/offset out of range"}, status_code=400
            )
        with lock:
            docs = list_documents(
                conn, category=category, tags=tags,
                since=since, limit=limit, offset=offset,
            )
        return JSONResponse([asdict(d) for d in docs])

    async def api_get_document(request: Request) -> JSONResponse:
        doc_id = request.path_params["doc_id"]
        include_text = request.query_params.get("include_text", "").lower() in (
            "1", "true", "yes"
        )
        with lock:
            doc = get_document(doc_id, conn, include_text=include_text)
        if doc is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(asdict(doc))

    # ---- job endpoints (require the queue to be present) ----------------

    async def api_upload(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        max_bytes = cfg.web.max_upload_mb * 1024 * 1024

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes * 2:
            # x2 for multipart overhead
            return JSONResponse(
                {"error": f"request exceeds {cfg.web.max_upload_mb} MB"},
                status_code=413,
            )

        form = await request.form()
        uploads = form.getlist("files")
        if not uploads:
            return JSONResponse(
                {"error": "no files field in form"}, status_code=400
            )

        category = (form.get("category") or "").strip() or None
        tags = _split_tags(form.get("tags") or "")

        job_ids: list[str] = []
        errors: list[dict] = []
        cfg.pending_uploads_dir.mkdir(parents=True, exist_ok=True)

        for upload in uploads:
            if not hasattr(upload, "read"):
                errors.append({
                    "filename": str(upload),
                    "reason": "not a file upload",
                })
                continue
            filename = _sanitize_filename(getattr(upload, "filename", "") or "")
            pending_id = str(ULID())
            pending_path = cfg.pending_uploads_dir / pending_id

            total = 0
            try:
                with pending_path.open("wb") as out:
                    while chunk := await upload.read(65536):
                        total += len(chunk)
                        if total > max_bytes:
                            out.close()
                            pending_path.unlink(missing_ok=True)
                            errors.append({
                                "filename": filename,
                                "reason": f"exceeds {cfg.web.max_upload_mb} MB cap",
                            })
                            break
                        out.write(chunk)
                    else:
                        job_id = jobs.enqueue_upload(
                            pending_path=pending_path,
                            filename=filename,
                            category=category,
                            tags=tags,
                            content_type_hint=getattr(upload, "content_type", None),
                        )
                        job_ids.append(job_id)
            finally:
                await upload.close()

        return JSONResponse({"job_ids": job_ids, "errors": errors})

    async def api_ingest_url(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        url = (body.get("url") or "").strip()
        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)
        if not url.startswith(("http://", "https://")):
            return JSONResponse(
                {"error": "url must be http(s)"}, status_code=400
            )
        category = (body.get("category") or "").strip() or None
        tags_field = body.get("tags") or []
        if isinstance(tags_field, str):
            tags = _split_tags(tags_field)
        else:
            tags = [str(t).strip() for t in tags_field if str(t).strip()]
        job_id = jobs.enqueue_url(url, category, tags)
        return JSONResponse({"job_id": job_id})

    async def api_list_jobs(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse([])
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            return JSONResponse({"error": "invalid limit"}, status_code=400)
        limit = max(1, min(200, limit))
        rows = jobs.list_recent(limit=limit)
        return JSONResponse([asdict(r) for r in rows])

    async def api_get_job(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        row = jobs.get(request.path_params["job_id"])
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(asdict(row))

    async def api_cancel_job(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        ok = jobs.cancel(request.path_params["job_id"])
        if not ok:
            return JSONResponse(
                {"error": "not cancellable (unknown, done, or already cancelled)"},
                status_code=409,
            )
        return JSONResponse({"ok": True})

    async def api_jobs_stream(request: Request) -> StreamingResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        jobs.broadcaster.subscribe(q, loop)

        async def gen():
            try:
                yield _sse_format("connected", {})
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield _sse_format(ev["event"], ev["job"])
            finally:
                jobs.broadcaster.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- curate endpoints ------------------------------------------------

    async def api_patch_document(request: Request) -> JSONResponse:
        doc_id = request.path_params["doc_id"]
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be an object"}, status_code=400)

        # Sentinel: Ellipsis means "field absent"; None means "unset".
        category: str | None | type(...) = ...
        if "category" in body:
            raw = body["category"]
            if raw is not None and not isinstance(raw, str):
                return JSONResponse({"error": "category must be a string or null"}, status_code=400)
            if isinstance(raw, str) and not raw.strip():
                category = None
            else:
                category = raw.strip() if isinstance(raw, str) else None

        tags: list[str] | None = None
        if "tags" in body:
            raw_tags = body["tags"]
            if isinstance(raw_tags, str):
                tags = _split_tags(raw_tags)
            elif isinstance(raw_tags, list):
                tags = [str(t).strip() for t in raw_tags if str(t).strip()]
            else:
                return JSONResponse({"error": "tags must be a list or comma-separated string"}, status_code=400)

        if category is ... and tags is None:
            return JSONResponse({"error": "no fields to update"}, status_code=400)

        with lock:
            ok = update_document_metadata(doc_id, conn, category=category, tags=tags)
        if not ok:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"ok": True})

    async def api_delete_document(request: Request) -> JSONResponse:
        doc_id = request.path_params["doc_id"]
        with lock:
            ok = delete_document(doc_id, conn, cfg)
        if not ok:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"deleted": True})

    async def api_rename_category(request: Request) -> JSONResponse:
        old = request.path_params["name"]
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        new = (body.get("to") or "").strip() if isinstance(body, dict) else ""
        if not new:
            return JSONResponse({"error": "'to' is required"}, status_code=400)
        with lock:
            n = rename_category(old, new, conn)
        return JSONResponse({"affected": n})

    async def api_delete_category(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        with lock:
            n = delete_category(name, conn)
        return JSONResponse({"affected": n})

    async def api_rename_tag(request: Request) -> JSONResponse:
        old = request.path_params["name"]
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        new = (body.get("to") or "").strip() if isinstance(body, dict) else ""
        if not new:
            return JSONResponse({"error": "'to' is required"}, status_code=400)
        with lock:
            n = rename_tag(old, new, conn)
        return JSONResponse({"affected": n})

    async def api_delete_tag(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        with lock:
            n = delete_tag(name, conn)
        return JSONResponse({"affected": n})

    return [
        Route("/api/info", api_info, methods=["GET"]),
        Route("/api/catalog", api_catalog, methods=["GET"]),
        Route("/api/search", api_search, methods=["GET"]),
        Route("/api/documents", api_list_documents, methods=["GET"]),
        Route("/api/documents/{doc_id}", api_get_document, methods=["GET"]),
        Route("/api/documents/{doc_id}", api_patch_document, methods=["PATCH"]),
        Route("/api/documents/{doc_id}", api_delete_document, methods=["DELETE"]),
        Route("/api/categories/{name}/rename", api_rename_category, methods=["POST"]),
        Route("/api/categories/{name}", api_delete_category, methods=["DELETE"]),
        Route("/api/tags/{name}/rename", api_rename_tag, methods=["POST"]),
        Route("/api/tags/{name}", api_delete_tag, methods=["DELETE"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/api/ingest-url", api_ingest_url, methods=["POST"]),
        Route("/api/jobs", api_list_jobs, methods=["GET"]),
        Route("/api/jobs/stream", api_jobs_stream, methods=["GET"]),
        Route("/api/jobs/{job_id}", api_get_job, methods=["GET"]),
        Route("/api/jobs/{job_id}/cancel", api_cancel_job, methods=["POST"]),
    ]
