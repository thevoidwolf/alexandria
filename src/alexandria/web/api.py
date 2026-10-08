"""JSON endpoints under /api/* for the browser UI (and curl scripting).

Also serves ``GET /files/<doc_id>``, the original content-addressed blob
route used by the "Open original" UI button and the ``get_original`` MCP
tool. That route lives here (not under /api/*) so it can be linked in
plain HTML and stitched into shareable signed URLs.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route
from ulid import ULID

from alexandria import anchors as _anchors
from alexandria.catalog import get_catalog, get_document, list_documents
from alexandria.classify import Suggestion, suggest_metadata, suggest_metadata_bulk
from alexandria.common import (
    anchor_payload,
    package_version,
    search_hit_payload,
    split_tags,
    suggestion_payload,
)
from alexandria.config import Config
from alexandria.curate import (
    delete_category,
    delete_document,
    delete_tag,
    rename_category,
    rename_tag,
    update_document_metadata,
)
from alexandria.originals import (
    blob_path,
    filename_from_source,
    lookup_blob_for,
    mime_for,
)
from alexandria.search import SEARCH_MODES, search
from alexandria.web.jobs import JobQueue

def _sanitize_filename(name: str) -> str:
    """Basename-only, non-empty, no leading dots."""
    name = (name or "").replace("\\", "/")
    name = Path(name).name.strip().lstrip(".")
    return name or "unnamed"


async def run_locked(lock: threading.Lock, fn, /, *args, **kwargs):
    """Run ``fn`` under ``lock`` on a worker thread.

    Handlers must not take the DB lock (or run embeddings) on the event loop:
    while another thread holds it, the whole server would stall.
    """
    def call():
        with lock:
            return fn(*args, **kwargs)

    return await run_in_threadpool(call)


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _rfc5987(value: str) -> str:
    """Percent-encode a filename for a Content-Disposition filename* param."""
    return quote(value, safe="")


def build_api_routes(
    cfg: Config,
    conn: sqlite3.Connection,
    lock: threading.Lock,
    jobs: JobQueue | None = None,
) -> list[Route]:
    async def api_info(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "version": package_version(),
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
        summary = await run_locked(lock, get_catalog, conn)
        return JSONResponse(asdict(summary))

    async def api_search(request: Request) -> JSONResponse:
        params = request.query_params
        query = (params.get("q") or "").strip()
        if not query:
            return JSONResponse([])
        category = params.get("category") or None
        tags = split_tags(params.get("tags"))
        try:
            limit = int(params.get("limit", "10"))
        except ValueError:
            return JSONResponse({"error": "invalid limit"}, status_code=400)
        if limit < 1 or limit > 100:
            return JSONResponse(
                {"error": "limit out of range (1..100)"}, status_code=400
            )
        mode = params.get("mode", "hybrid")
        if mode not in SEARCH_MODES:
            return JSONResponse(
                {"error": f"invalid mode: {mode!r}"}, status_code=400
            )
        hits = await run_locked(
            lock, search, query, conn, cfg,
            category=category, tags=tags, limit=limit, mode=mode,
        )
        return JSONResponse([search_hit_payload(h) for h in hits])

    async def api_list_documents(request: Request) -> JSONResponse:
        params = request.query_params
        category = params.get("category") or None
        tags = split_tags(params.get("tags"))
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
        docs = await run_locked(
            lock, list_documents, conn,
            category=category, tags=tags, since=since, limit=limit, offset=offset,
        )
        return JSONResponse([asdict(d) for d in docs])

    async def api_get_document(request: Request) -> JSONResponse:
        doc_id = request.path_params["doc_id"]
        include_text = request.query_params.get("include_text", "").lower() in (
            "1", "true", "yes"
        )
        doc = await run_locked(
            lock, get_document, doc_id, conn, include_text=include_text
        )
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

        # Starlette parses (and spools to disk) the whole multipart body
        # before the per-file cap below runs, so bound the request up front.
        # Requiring Content-Length rules out unbounded chunked uploads; the
        # server stops reading at the declared length.
        try:
            content_length = int(request.headers["content-length"])
        except KeyError:
            return JSONResponse(
                {"error": "Content-Length required"}, status_code=411
            )
        except ValueError:
            return JSONResponse(
                {"error": "invalid Content-Length"}, status_code=400
            )
        if content_length > max_bytes * 2:
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
        tags = split_tags(form.get("tags"))

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
            too_big = False
            try:
                with pending_path.open("wb") as out:
                    while chunk := await upload.read(65536):
                        total += len(chunk)
                        if total > max_bytes:
                            too_big = True
                            break
                        out.write(chunk)
            finally:
                await upload.close()

            if too_big:
                pending_path.unlink(missing_ok=True)
                errors.append({
                    "filename": filename,
                    "reason": f"exceeds {cfg.web.max_upload_mb} MB cap",
                })
                continue
            # Enqueue only after the file is closed: the worker may pick the
            # job up immediately and must see the complete upload.
            job_id = await run_in_threadpool(
                jobs.enqueue_upload,
                pending_path=pending_path,
                filename=filename,
                category=category,
                tags=tags,
                content_type_hint=getattr(upload, "content_type", None),
            )
            job_ids.append(job_id)

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
            tags = split_tags(tags_field)
        else:
            tags = [str(t).strip() for t in tags_field if str(t).strip()]
        job_id = await run_in_threadpool(jobs.enqueue_url, url, category, tags)
        return JSONResponse({"job_id": job_id})

    async def api_list_jobs(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse([])
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            return JSONResponse({"error": "invalid limit"}, status_code=400)
        limit = max(1, min(200, limit))
        rows = await run_in_threadpool(jobs.list_recent, limit=limit)
        return JSONResponse([asdict(r) for r in rows])

    async def api_get_job(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        row = await run_in_threadpool(jobs.get, request.path_params["job_id"])
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(asdict(row))

    async def api_cancel_job(request: Request) -> JSONResponse:
        if jobs is None:
            return JSONResponse(
                {"error": "job queue disabled"}, status_code=503
            )
        ok = await run_in_threadpool(jobs.cancel, request.path_params["job_id"])
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
                tags = split_tags(raw_tags)
            elif isinstance(raw_tags, list):
                tags = [str(t).strip() for t in raw_tags if str(t).strip()]
            else:
                return JSONResponse({"error": "tags must be a list or comma-separated string"}, status_code=400)

        if category is ... and tags is None:
            return JSONResponse({"error": "no fields to update"}, status_code=400)

        ok = await run_locked(
            lock, update_document_metadata, doc_id, conn,
            category=category, tags=tags,
        )
        if not ok:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"ok": True})

    async def api_delete_document(request: Request) -> JSONResponse:
        doc_id = request.path_params["doc_id"]
        ok = await run_locked(lock, delete_document, doc_id, conn, cfg)
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
        n = await run_locked(lock, rename_category, old, new, conn)
        return JSONResponse({"affected": n})

    async def api_delete_category(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        n = await run_locked(lock, delete_category, name, conn)
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
        n = await run_locked(lock, rename_tag, old, new, conn)
        return JSONResponse({"affected": n})

    async def api_delete_tag(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        n = await run_locked(lock, delete_tag, name, conn)
        return JSONResponse({"affected": n})

    # ---- classifier ------------------------------------------------------

    async def api_suggest_metadata(request: Request) -> JSONResponse:
        doc_id = request.path_params["doc_id"]
        apply_writes = False
        # Custom category/tag overrides let the UI accept a partial subset
        # of the classifier's suggestion (uncheck bad tags before applying).
        override_category: str | None | type(...) = ...
        override_tags: list[str] | None = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                apply_writes = bool(body.get("apply", False))
                if "category" in body:
                    raw = body["category"]
                    if raw is None or raw == "":
                        override_category = None
                    elif isinstance(raw, str):
                        override_category = raw.strip() or None
                if "tags" in body and isinstance(body["tags"], list):
                    override_tags = [str(t).strip() for t in body["tags"] if str(t).strip()]
        except (ValueError, json.JSONDecodeError):
            apply_writes = (request.query_params.get("apply") or "").lower() in (
                "1", "true", "yes"
            )

        def suggest_and_apply() -> tuple[Suggestion | None, bool]:
            s = suggest_metadata(doc_id, conn, cfg)
            if s is None or not apply_writes:
                return s, False
            cat = override_category if override_category is not ... else (
                s.suggested_category if s.suggested_category else ...
            )
            tags = override_tags if override_tags is not None else (
                s.suggested_tags if s.suggested_tags else None
            )
            if cat is ... and tags is None:
                return s, False
            update_document_metadata(doc_id, conn, category=cat, tags=tags)
            return s, True

        s, applied = await run_locked(lock, suggest_and_apply)
        if s is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(suggestion_payload(s, applied))

    # ---- anchors --------------------------------------------------------

    async def api_list_anchors(_request: Request) -> JSONResponse:
        items = await run_locked(lock, _anchors.list_anchors, conn)
        return JSONResponse([anchor_payload(a) for a in items])

    async def api_set_anchor(request: Request) -> JSONResponse:
        kind = request.path_params["kind"]
        name = request.path_params["name"]
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be an object"}, status_code=400)
        description = str(body.get("description") or "").strip()
        if not description:
            return JSONResponse(
                {"error": "description is required"}, status_code=400
            )
        try:
            a = await run_locked(
                lock, _anchors.set_anchor, conn, cfg, kind, name, description
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(anchor_payload(a))

    async def api_delete_anchor(request: Request) -> JSONResponse:
        kind = request.path_params["kind"]
        name = request.path_params["name"]
        ok = await run_locked(lock, _anchors.delete_anchor, conn, kind, name)
        return JSONResponse({"deleted": ok})

    async def api_import_anchors(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        items = body.get("anchors") if isinstance(body, dict) else body
        if not isinstance(items, list):
            return JSONResponse(
                {"error": "expected {\"anchors\": [...]} or a top-level list"},
                status_code=400,
            )
        r = await run_locked(lock, _anchors.import_anchors, conn, cfg, items)
        return JSONResponse(r)

    async def api_export_anchors(_request: Request) -> JSONResponse:
        items = await run_locked(lock, _anchors.export_anchors, conn)
        return JSONResponse({"anchors": items})

    async def api_suggest_metadata_bulk(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        try:
            limit = int(body.get("limit", 20))
            offset = int(body.get("offset", 0))
        except (TypeError, ValueError):
            return JSONResponse({"error": "invalid limit/offset"}, status_code=400)
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        missing_only = bool(body.get("missing_only", True))
        apply_writes = bool(body.get("apply", False))

        r = await run_locked(
            lock, suggest_metadata_bulk, conn, cfg,
            limit=limit, offset=offset,
            missing_only=missing_only, apply=apply_writes,
        )
        return JSONResponse({
            "scanned": r.scanned,
            "suggestions": [suggestion_payload(s) for s in r.suggestions],
            "applied": list(r.applied),
        })

    # ---- original-file endpoint ----------------------------------------

    async def api_get_original(request: Request) -> "JSONResponse | FileResponse":
        doc_id = request.path_params["doc_id"]
        download = (request.query_params.get("download") or "").lower() in (
            "1", "true", "yes"
        )
        info = await run_locked(lock, lookup_blob_for, conn, doc_id)
        if info is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        sha, ctype, src_uri = info

        if not cfg.storage.keep_blobs:
            return JSONResponse(
                {"error": "original not retained (storage.keep_blobs = false)"},
                status_code=404,
            )
        blob = blob_path(cfg, sha)
        if not blob.exists():
            return JSONResponse(
                {"error": "original blob missing on disk"},
                status_code=404,
            )

        filename = filename_from_source(src_uri, ctype, doc_id)
        disposition = "attachment" if download else "inline"
        # RFC 5987: filename* for non-ASCII names; plain filename is a fallback
        # for ancient clients. Starlette's FileResponse sets Content-Disposition
        # itself, so we bypass it by using the low-level headers path.
        safe_ascii = filename.encode("ascii", "replace").decode("ascii").replace('"', "")
        headers = {
            "Content-Disposition": (
                f'{disposition}; filename="{safe_ascii}"; '
                f"filename*=UTF-8''{_rfc5987(filename)}"
            ),
            "Cache-Control": "private, max-age=0, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        }
        # Originals are untrusted content (any fetched web page) served from
        # the app's own origin. ``sandbox`` gives them an opaque origin so a
        # stored page's scripts can't ride the session cookie into /api/*.
        # PDFs are exempt: Chromium refuses to render PDFs in a sandboxed
        # document, and its viewer doesn't run page script on our origin.
        if ctype != "pdf":
            headers["Content-Security-Policy"] = "sandbox"
        return FileResponse(
            path=str(blob),
            media_type=mime_for(ctype),
            headers=headers,
        )

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
        Route("/api/documents/{doc_id}/suggest-metadata",
              api_suggest_metadata, methods=["POST"]),
        Route("/api/suggest-metadata-bulk",
              api_suggest_metadata_bulk, methods=["POST"]),
        Route("/api/anchors", api_list_anchors, methods=["GET"]),
        Route("/api/anchors/import", api_import_anchors, methods=["POST"]),
        Route("/api/anchors/export", api_export_anchors, methods=["GET"]),
        Route("/api/anchors/{kind}/{name}", api_set_anchor, methods=["PUT"]),
        Route("/api/anchors/{kind}/{name}", api_delete_anchor, methods=["DELETE"]),
        Route("/files/{doc_id}", api_get_original, methods=["GET"]),
    ]
