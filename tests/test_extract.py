import importlib.util
import os
from pathlib import Path

import pytest

from alexandria.config import (
    Config,
    ExtractorsConfig,
    PdfExtractorConfig,
)
from alexandria.ingest.extract import detect_content_type, extract

FIXTURES = Path(__file__).parent / "fixtures"


def test_detects_html_from_magic_bytes():
    data = b"<!doctype html>\n<html></html>"
    assert detect_content_type(data, None, None) == "html"


def test_detects_html_from_body_start():
    # No doctype, starts with <html or <body
    data = b"<html><body>x</body></html>"
    assert detect_content_type(data, None, None) == "html"


def test_detects_pdf_from_magic():
    assert detect_content_type(b"%PDF-1.4\n...", None, None) == "pdf"


def test_falls_back_to_extension_for_markdown():
    data = b"# just a header\n\nno signals\n"
    assert detect_content_type(data, None, "note.md") == "md"


def test_defaults_to_txt_without_signals():
    data = b"plain content here"
    assert detect_content_type(data, None, "unknown.dat") == "txt"


def test_content_type_hint_is_honored():
    # Ambiguous body, but the HTTP hint disambiguates.
    data = b"some plain-ish content"
    assert detect_content_type(data, "application/pdf", None) == "pdf"


def test_extract_plain_text_leaves_content_intact():
    data = FIXTURES.joinpath("plain.txt").read_bytes()
    result = extract(data, "txt")
    assert "electricity bill" in result.text
    assert result.extractor == "plaintext"


def test_extract_html_strips_boilerplate_and_pulls_title():
    data = FIXTURES.joinpath("page.html").read_bytes()
    result = extract(data, "html")
    assert result.title == "On knowledge stores"
    assert "menu junk" not in result.text        # nav stripped
    assert "hybrid retrieval" in result.text.lower()
    assert result.extractor == "trafilatura"


def test_extract_normalizes_whitespace_and_newlines():
    data = b"one\r\n\r\n\r\n\r\nfour     spaces\tand\ttabs\n"
    result = extract(data, "txt")
    assert "\r" not in result.text
    assert "\n\n\n" not in result.text            # collapsed
    assert "four spaces and tabs" in result.text  # tabs/multi-spaces collapsed


def test_extract_strips_zero_width_chars():
    # zero-width space between "he" and "llo"
    data = "he​llo world".encode("utf-8")
    result = extract(data, "txt")
    assert result.text.startswith("hello world")


def test_detects_pdf_from_real_file():
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    assert detect_content_type(data, None, None) == "pdf"


def test_extract_pdf_pulls_body_text_and_creation_date():
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    result = extract(data, "pdf")

    assert result.content_type == "pdf"
    assert result.extractor == "pypdf"                    # pdfplumber fallback not triggered

    # Body text survives from a multi-page, LaTeX-generated PDF with math + figures.
    assert "Observing Electric Currents in Space" in result.text
    assert "Michael Clarage" in result.text

    # This PDF has no /Title (typical of pdfTeX output) but does carry /CreationDate.
    assert result.title is None
    assert result.published_at is not None
    assert "2025" in result.published_at

    # Normalization still applied to PDF output.
    assert "\r" not in result.text
    assert "\n\n\n" not in result.text


# Marker downloads ~2GB of weights and is CPU-glacial (~40 min on a 6-page paper).
# Gated: skipped unless marker is installed AND the caller explicitly opts in.
# On the GPU server, run:  ALEXANDRIA_TEST_MARKER=1 uv run pytest -k marker
_MARKER_AVAILABLE = importlib.util.find_spec("marker") is not None


@pytest.mark.skipif(
    not _MARKER_AVAILABLE or os.environ.get("ALEXANDRIA_TEST_MARKER") != "1",
    reason="marker backend test disabled; set ALEXANDRIA_TEST_MARKER=1 to enable",
)
def test_extract_pdf_marker_backend_round_trips_latex(tmp_path: Path):
    cfg = Config(
        home=tmp_path,
        extractors=ExtractorsConfig(
            pdf=PdfExtractorConfig(backend="marker", device="auto"),
        ),
    )
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    result = extract(data, "pdf", cfg)

    assert result.extractor.startswith("marker@")

    # Math from paper.pdf survives as LaTeX (pypdf couldn't reconstruct any of these).
    assert "$$" in result.text                    # display math delimiter present
    assert r"\frac" in result.text                # fraction bar recovered (equation 1)
    assert r"\alpha" in result.text               # α as LaTeX macro
    assert r"\vec" in result.text                 # vector arrows (equation 2)
    assert "_{B_z}" in result.text or "B_z" in result.text  # subscript recovered
