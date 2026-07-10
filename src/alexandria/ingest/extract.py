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

# Greek letters + common math operators. Prose sprinkles the odd Greek ("alpha
# level was 0.05") but math-heavy docs blow past ~1/1000 chars easily.
_MATH_CHARS = re.compile(
    r"[αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
    r"∫∑∇∂×·⋅∞≈≤≥±→⇒∈∀∃∪∩⊂⊃∅ℝℂℕℤℚ]"
)

# LaTeX / math-typesetter font-name substrings (matched case-insensitively).
# Presence of any of these in a PDF's font resource dict is a near-certain
# "math paper" signal. pdfTeX PDFs subset-prefix names ("VRQZFR+rtxmi"), so we
# substring-match.
_MATH_FONT_MARKERS = (
    "cmex", "cmmi", "cmsy", "cmbsy", "cmmib",    # Computer Modern math series
    "lmmath", "lmroman",                          # Latin Modern (lmmath variants)
    "stixmath", "stixsize", "stix-",              # STIX math
    "xitsmath",                                   # XITS math
    "txmi", "txsy", "txex",                       # TX Fonts (LaTeX txfonts)
    "rtxmi", "rtxsy",                             # RTX (rtxfonts) — physics papers
    "mtsyn", "mtsy",                              # MathTime symbol
    "mathjax",                                    # MathJax-exported PDFs
)

# Markdown emphasis wrappers around marker's extracted titles.
_MD_EMPHASIS = re.compile(r"\*{1,3}([^\n*]+)\*{1,3}")


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


_SURYA_BATCH_VARS = (
    "RECOGNITION_BATCH_SIZE",
    "DETECTOR_BATCH_SIZE",
    "LAYOUT_BATCH_SIZE",
    "OCR_ERROR_BATCH_SIZE",
    "TABLE_REC_BATCH_SIZE",
)


def _extract_pdf_marker(
    data: bytes,
    device: str,
    batch_size: int = 0,
    cooldown_seconds: float = 0.0,
) -> tuple[str, str | None, str | None, str | None, str]:
    # marker's CLI sets TORCH_DEVICE up-front; the Python API respects it too.
    os.environ.setdefault("TORCH_DEVICE", _resolve_device(device))

    # Cap surya's per-phase batch sizes for thermal safety on small GPUs.
    # Must happen before the import — surya reads these at module load.
    if batch_size > 0:
        for var in _SURYA_BATCH_VARS:
            os.environ[var] = str(batch_size)

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
    # source PDF has no /Title metadata (typical for pdfTeX papers). Strip any
    # markdown emphasis wrappers ("# **Foo**" → "Foo").
    title: str | None = None
    for line in text.split("\n", 200):
        s = line.strip()
        if s.startswith("# "):
            candidate = s[2:].strip()
            title = _MD_EMPHASIS.sub(r"\1", candidate).strip() or None
            break

    try:
        from importlib.metadata import version as _pkg_version
        version = _pkg_version("marker-pdf")
    except Exception:
        version = "unknown"

    # Post-run cooldown: lets the card breathe before the next doc arrives.
    # Costs nothing on single-doc ingest; matters for folder walks.
    if cooldown_seconds > 0:
        import time
        time.sleep(cooldown_seconds)

    return text, title, None, None, f"marker@{version}"


# ---- adaptive dispatch (M9) --------------------------------------------------


def _math_density(text: str) -> float:
    """Return math-symbol count per 1000 chars. 0.0 for empty text."""
    if not text:
        return 0.0
    return len(_MATH_CHARS.findall(text)) / len(text) * 1000.0


def _has_math_fonts(data: bytes) -> tuple[bool, str]:
    """Return (True, matching-font-name) if the PDF embeds a LaTeX-family math
    font, else (False, ""). Runs in ~10ms — pypdf only walks the font dicts,
    not the page content."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            fonts = ((page.get("/Resources") or {}).get("/Font") or {}) if page else {}
            for fkey in fonts:
                try:
                    base = str(fonts[fkey].get("/BaseFont") or "")
                except Exception:
                    continue
                base_lower = base.lower()
                for m in _MATH_FONT_MARKERS:
                    if m in base_lower:
                        return True, base
    except Exception:
        # Malformed PDFs are marker's job anyway; err on the side of upgrading.
        return False, ""
    return False, ""


def _detect_math_needed(
    data: bytes, pypdf_text: str, threshold: float
) -> tuple[bool, str]:
    """Decide whether an auto-mode PDF should be upgraded from pypdf → marker.

    Returns (needs_marker, reason) where reason is human-readable for logging.
    """
    # Signal: pypdf produced no text (probably a scan) → marker's surya OCR.
    if not pypdf_text.strip():
        return True, "empty pypdf text (scan?)"

    # Signal: LaTeX/math font embedded in the PDF.
    has_fonts, font_name = _has_math_fonts(data)
    if has_fonts:
        return True, f"math font: {font_name}"

    # Signal: raw math-symbol density in pypdf's flattened text.
    density = _math_density(pypdf_text)
    if density >= threshold:
        return True, f"symbol density {density:.2f}/1000 ≥ {threshold}"

    return False, f"symbol density {density:.2f}/1000 < {threshold}, no math fonts"


def _extract_pdf(
    data: bytes, cfg: Config | None
) -> tuple[str, str | None, str | None, str | None, str]:
    backend = "pypdf"
    device = "auto"
    threshold = 2.0
    batch_size = 0
    cooldown = 0.0
    if cfg is not None:
        pdf_cfg = cfg.extractors.pdf
        backend = pdf_cfg.backend
        device = pdf_cfg.device
        threshold = pdf_cfg.math_symbol_threshold
        batch_size = pdf_cfg.marker_batch_size
        cooldown = pdf_cfg.marker_cooldown_seconds

    if backend == "marker":
        try:
            return _extract_pdf_marker(data, device, batch_size, cooldown)
        except Exception as e:
            log.warning("marker extraction failed (%s); falling back to pypdf", e)
        return _extract_pdf_pypdf(data)

    if backend == "auto":
        # Run pypdf first (cheap). Consult the two signals against its output.
        # If math-heavy or empty, upgrade to marker; otherwise keep pypdf.
        pypdf_result = _extract_pdf_pypdf(data)
        pypdf_text = pypdf_result[0]
        needs_marker, reason = _detect_math_needed(data, pypdf_text, threshold)
        if needs_marker:
            log.info("auto: upgrading to marker (%s)", reason)
            try:
                return _extract_pdf_marker(data, device, batch_size, cooldown)
            except Exception as e:
                log.warning(
                    "marker upgrade failed (%s); keeping pypdf output", e
                )
                return pypdf_result
        log.info("auto: keeping pypdf (%s)", reason)
        return pypdf_result

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
