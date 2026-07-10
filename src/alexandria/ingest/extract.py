from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

import filetype

if TYPE_CHECKING:
    from alexandria.config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Extracted:
    content_type: str      # "pdf" | "html" | "txt" | "md"
    text: str              # normalized text
    title: str | None
    author: str | None
    published_at: str | None
    extractor: str         # e.g. "pypdf@5.1"


_HTML_MAGIC = re.compile(rb"^\s*(<!doctype html|<html|<head|<body)", re.IGNORECASE)
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def detect_content_type(data: bytes, hint: str | None, filename: str | None) -> str:
    """Return one of pdf|html|txt|md by sniffing bytes, then falling back to filename."""
    kind = filetype.guess(data)
    if kind is not None:
        mime = kind.mime
        if mime == "application/pdf":
            return "pdf"
        if mime.startswith("text/html") or mime == "application/xhtml+xml":
            return "html"
    if _HTML_MAGIC.match(data[:512]):
        return "html"
    if hint:
        h = hint.lower()
        if "pdf" in h:
            return "pdf"
        if "html" in h:
            return "html"
        if "markdown" in h:
            return "md"
    if filename:
        low = filename.lower()
        if low.endswith(".pdf"):
            return "pdf"
        if low.endswith((".html", ".htm")):
            return "html"
        if low.endswith((".md", ".markdown")):
            return "md"
    return "txt"


def _extract_pdf_pypdf(data: bytes) -> tuple[str, str | None, str | None, str | None, str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(pages).strip()
    extractor = "pypdf"

    if not text.strip():
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n\n".join((p.extract_text() or "") for p in pdf.pages).strip()
        extractor = "pdfplumber"

    meta = reader.metadata or {}
    title = (meta.get("/Title") or None) if meta else None
    author = (meta.get("/Author") or None) if meta else None
    published = None
    if meta and (raw := meta.get("/CreationDate")):
        published = str(raw)
    return text, title, author, published, extractor


# Marker's model dict is expensive to build (loads ~2GB of weights). Cache it on
# the module so a long-running server pays the cost once, not per-document.
_MARKER_MODELS: dict | None = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _extract_pdf_marker(
    data: bytes, device: str
) -> tuple[str, str | None, str | None, str | None, str]:
    # marker's CLI sets TORCH_DEVICE up-front; the Python API respects it too.
    os.environ.setdefault("TORCH_DEVICE", _resolve_device(device))

    import marker
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    global _MARKER_MODELS
    if _MARKER_MODELS is None:
        _MARKER_MODELS = create_model_dict()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(data)
        tmp_path = tf.name
    try:
        converter = PdfConverter(artifact_dict=_MARKER_MODELS)
        rendered = converter(tmp_path)
        text, _, _images = text_from_rendered(rendered)
    finally:
        os.unlink(tmp_path)

    # marker output is markdown; first h1 is a reasonable title fallback when the
    # source PDF has no /Title metadata (typical for pdfTeX papers).
    title: str | None = None
    for line in text.split("\n", 200):
        s = line.strip()
        if s.startswith("# "):
            title = s[2:].strip()
            break

    version = getattr(marker, "__version__", None) or "unknown"
    return text, title, None, None, f"marker@{version}"


def _extract_pdf(
    data: bytes, cfg: Config | None
) -> tuple[str, str | None, str | None, str | None, str]:
    backend = "pypdf"
    device = "auto"
    if cfg is not None:
        backend = cfg.extractors.pdf.backend
        device = cfg.extractors.pdf.device

    if backend == "marker":
        try:
            return _extract_pdf_marker(data, device)
        except Exception as e:
            log.warning("marker extraction failed (%s); falling back to pypdf", e)
    return _extract_pdf_pypdf(data)


def _extract_html(data: bytes) -> tuple[str, str | None, str | None, str | None, str]:
    import trafilatura

    raw = data.decode("utf-8", errors="replace")
    text = trafilatura.extract(raw, include_comments=False, include_tables=True) or ""
    title = None
    author = None
    published = None
    try:
        meta = trafilatura.extract_metadata(raw)
        if meta is not None:
            title = meta.title or None
            author = meta.author or None
            published = meta.date or None
    except Exception:
        pass
    return text, title, author, published, "trafilatura"


def _extract_text(data: bytes) -> tuple[str, str | None, str | None, str | None, str]:
    return data.decode("utf-8", errors="replace"), None, None, None, "plaintext"


def extract(
    data: bytes, content_type: str, cfg: Config | None = None
) -> Extracted:
    if content_type == "pdf":
        text, title, author, published, extractor = _extract_pdf(data, cfg)
    elif content_type == "html":
        text, title, author, published, extractor = _extract_html(data)
    else:  # txt, md
        text, title, author, published, extractor = _extract_text(data)

    def _norm_meta(v: str | None) -> str | None:
        if v is None:
            return None
        # Titles/authors/dates: collapse all whitespace (including newlines) into single spaces.
        v = unicodedata.normalize("NFC", v)
        v = _ZERO_WIDTH.sub("", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v or None

    return Extracted(
        content_type=content_type,
        text=_normalize(text),
        title=_norm_meta(title),
        author=_norm_meta(author),
        published_at=_norm_meta(published),
        extractor=extractor,
    )
