import importlib.util
import os
from pathlib import Path

import pytest

from alexandria.config import (
    Config,
    ExtractorsConfig,
    PdfExtractorConfig,
)
from alexandria.ingest.extract import (
    _detect_math_needed,
    _has_math_fonts,
    _math_density,
    detect_content_type,
    extract,
)

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


# ---- M11: math-friendly text cleaner ----------------------------------------


def test_normalize_folds_superscript_digits_via_nfkc():
    # ² and ¹¹ (superscript) should fold to ASCII digits so search matches.
    data = "energy ~ 10² joules; current 10¹¹ amperes".encode("utf-8")
    result = extract(data, "txt")
    assert "10^2" not in result.text  # not what NFKC produces; ensure no false shape
    assert "102 joules" in result.text
    assert "1011 amperes" in result.text


def test_normalize_expands_ligatures():
    # LaTeX PDFs commonly emit ﬁ ﬀ ﬂ as single codepoints. Search for "efficient"
    # must match text stored as "eﬃcient".
    data = "eﬃcient conﬂict aﬀordable ﬁnancial".encode("utf-8")
    result = extract(data, "txt")
    assert "efficient conflict affordable financial" in result.text


def test_normalize_repairs_hyphenated_linebreaks():
    # "infor-\nmation" is unsearchable as "information" without repair.
    data = b"the infor-\nmation was found in the docu-\nment"
    result = extract(data, "txt")
    assert "information" in result.text
    assert "document" in result.text
    assert "infor-" not in result.text


def test_normalize_strips_soft_hyphens():
    # Soft hyphen U+00AD is invisible but breaks tokenization.
    data = "opti­mize search­able tokeni­zation".encode("utf-8")
    result = extract(data, "txt")
    assert "optimize" in result.text
    assert "searchable" in result.text
    assert "­" not in result.text


def test_normalize_form_feed_becomes_paragraph_break():
    data = b"end of page one\fbeginning of page two"
    result = extract(data, "txt")
    assert "end of page one\n\nbeginning of page two" in result.text
    assert "\f" not in result.text


def test_normalize_strips_lone_surrogates_and_control_chars():
    # Lone surrogate + bell (0x07) + DEL (0x7f) — pypdf occasionally leaks these
    # from broken PDF ToUnicode tables. All must be stripped.
    raw = "clean\ud800 text\x07 with\x7f junk"
    data = raw.encode("utf-8", errors="surrogatepass")
    result = extract(data, "txt")
    assert "clean" in result.text
    assert "\ud800" not in result.text
    assert "\x07" not in result.text
    assert "\x7f" not in result.text


def test_detects_pdf_from_real_file():
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    assert detect_content_type(data, None, None) == "pdf"


def test_extract_pdf_pulls_body_text_and_creation_date():
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    result = extract(data, "pdf")

    assert result.content_type == "pdf"
    assert result.extractor == "pypdf"                    # pdfplumber fallback not triggered

    # Body text survives from a multi-page, LaTeX-generated PDF with math.
    assert "A Sample Paper on Vector Fields" in result.text
    assert "Ada Fixtura" in result.text

    # This PDF has no /Title (typical of pdfTeX output), so the title comes
    # from its first page. It does carry /CreationDate.
    assert result.title == "A Sample Paper on Vector Fields"
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


# ---- M9: adaptive backend detection -----------------------------------------


def test_math_density_zero_for_plain_prose():
    text = (
        "This is a normal paragraph of English text with no unusual symbols, "
        "just words describing an ordinary situation on a Tuesday afternoon."
    )
    assert _math_density(text) == 0.0


def test_math_density_high_for_equation_soup():
    # Handful of Greek + operators packed into a short string.
    text = "∫α + β · γ dx = δ; ∑ε_n → ∞ ≤ π"
    assert _math_density(text) > 20.0    # very dense — well above the 1.0 threshold


def test_math_density_low_for_prose_with_occasional_greek():
    # Realistic prose length (~3000 chars) with one Greek mention — the
    # "alpha level was 0.05" style. Should sit well under the 2.0 threshold.
    prose = (
        "The study followed a standard randomized-controlled design across "
        "three cohorts recruited from partner clinics over the fiscal year. "
        "Participants provided informed consent and completed the baseline "
        "questionnaire in a supervised setting; missing responses were "
        "handled via multiple imputation with ten iterations. The α level "
        "was set to 0.05 for all pre-registered hypotheses, with a "
        "Bonferroni correction applied for the secondary analyses. "
        "Effect sizes are reported with 95% confidence intervals following "
        "the recommendations of the reporting checklist. Adherence to the "
        "protocol was verified by an independent monitor at each site. "
    ) * 4
    density = _math_density(prose)
    assert density < 2.0, f"expected prose density < 2.0, got {density:.2f}"


def test_math_density_handles_empty_string():
    assert _math_density("") == 0.0


def test_has_math_fonts_detects_paper_pdf():
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    found, font_name = _has_math_fonts(data)
    assert found is True
    # paper.pdf is built with default Computer Modern, so its math embeds the
    # cmmi/cmsy/cmex family. Assert on the broader family list our detector
    # supports (rtxfonts/txfonts show up in other physics papers).
    lower = font_name.lower()
    assert any(m in lower for m in ("cmmi", "cmsy", "cmex", "rtxmi", "txsy", "txmi"))


def test_detect_math_needed_positive_on_paper_pdf():
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    # Use pypdf's text as the density input (real auto-mode flow).
    from alexandria.ingest.extract import _extract_pdf_pypdf
    text, *_ = _extract_pdf_pypdf(data)
    needs, reason = _detect_math_needed(data, text, threshold=1.0)
    assert needs is True
    assert reason  # non-empty explanation


def test_detect_math_needed_escalates_empty_text():
    needs, reason = _detect_math_needed(b"%PDF-1.4\n%stub", "", threshold=1.0)
    assert needs is True
    assert "empty" in reason.lower()


def test_marker_batch_size_sets_env_vars(monkeypatch):
    """batch_size > 0 must set every surya BATCH env var *before* marker import."""
    from alexandria.ingest.extract import _SURYA_BATCH_VARS, _extract_pdf_marker

    # Clear env so we see fresh writes.
    for var in _SURYA_BATCH_VARS:
        monkeypatch.delenv(var, raising=False)

    # Stub the marker import chain so we don't actually load 2GB of weights.
    fake_pdf = type("P", (), {"__init__": lambda self, *a, **k: None,
                              "__call__": lambda self, path: object()})

    def fake_output(rendered):
        return ("stub text", None, [])

    def fake_create():
        return {}

    import sys
    import types
    for name, mod in {
        "marker": types.ModuleType("marker"),
        "marker.converters": types.ModuleType("marker.converters"),
        "marker.converters.pdf": types.ModuleType("marker.converters.pdf"),
        "marker.models": types.ModuleType("marker.models"),
        "marker.output": types.ModuleType("marker.output"),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["marker.converters.pdf"].PdfConverter = fake_pdf
    sys.modules["marker.models"].create_model_dict = fake_create
    sys.modules["marker.output"].text_from_rendered = fake_output

    # Ensure a fresh temp file gets written; marker import is stubbed so
    # the converter call above is a no-op stub.
    _extract_pdf_marker(b"%PDF-stub", device="cpu", batch_size=3)

    for var in _SURYA_BATCH_VARS:
        assert os.environ.get(var) == "3", f"{var} not set from batch_size"


def test_marker_cooldown_calls_sleep(monkeypatch):
    """cooldown_seconds > 0 must sleep before returning."""
    from alexandria.ingest.extract import _extract_pdf_marker

    calls = []
    import sys
    import types
    import time as _time_mod
    for name, mod in {
        "marker": types.ModuleType("marker"),
        "marker.converters": types.ModuleType("marker.converters"),
        "marker.converters.pdf": types.ModuleType("marker.converters.pdf"),
        "marker.models": types.ModuleType("marker.models"),
        "marker.output": types.ModuleType("marker.output"),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["marker.converters.pdf"].PdfConverter = type(
        "P", (), {"__init__": lambda self, *a, **k: None,
                  "__call__": lambda self, path: object()}
    )
    sys.modules["marker.models"].create_model_dict = lambda: {}
    sys.modules["marker.output"].text_from_rendered = lambda r: ("", None, [])

    monkeypatch.setattr(_time_mod, "sleep", lambda s: calls.append(s))

    _extract_pdf_marker(b"%PDF-stub", device="cpu", cooldown_seconds=1.5)
    assert 1.5 in calls, f"expected sleep(1.5), got {calls}"


def test_auto_mode_stays_on_pypdf_for_math_pdf_after_m11(tmp_path: Path, caplog):
    """M11: auto no longer routes math-heavy docs to marker. It stays on pypdf
    (with the M11 cleaner) and logs the math signal as an advisory."""
    import logging

    cfg = Config(
        home=tmp_path,
        extractors=ExtractorsConfig(
            pdf=PdfExtractorConfig(backend="auto", device="auto"),
        ),
    )
    data = FIXTURES.joinpath("paper.pdf").read_bytes()
    with caplog.at_level(logging.INFO, logger="alexandria.ingest.extract"):
        result = extract(data, "pdf", cfg)

    assert result.extractor == "pypdf"
    # The math-density / math-font signal should have been logged with the
    # "math-heavy document" advisory so operators know why the doc looked odd.
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "math-heavy" in joined


# ---- titles when the source's metadata has none ----------------------------


def _pdf(content: str, title: str | None = None) -> bytes:
    """A one-page Letter-size PDF whose page runs ``content`` (PDF text
    operators, Helvetica as /F1), with /Title metadata if given."""
    stream = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]
    if title is not None:
        objects.append(b"<< /Title (%s) >>" % title.encode("latin-1"))
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    info = b" /Info 6 0 R" if title is not None else b""
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\n" % (len(objects) + 1, info)
    return out + b"startxref\n%d\n%%%%EOF\n" % xref


def _line(text: str, size: float, x: float, y: float) -> str:
    return f"BT /F1 {size} Tf {x} {y} Td ({text}) Tj ET\n"


def test_pdf_title_metadata_wins_over_the_first_page():
    data = _pdf(_line("A Large Heading", 20, 72, 700), title="The Metadata Title")
    assert extract(data, "pdf").title == "The Metadata Title"


def test_pdf_title_falls_back_when_the_metadata_title_is_blank():
    data = _pdf(_line("The Real Title", 16, 72, 700), title="   ")
    assert extract(data, "pdf").title == "The Real Title"

def test_pdf_title_ignores_rotated_margin_text():
    # arXiv stamps its identifier up the left margin, rotated and in a large font.
    stamp = "BT /F1 20 Tf 0 1 -1 0 40 150 Tm (arXiv:2401.00001v1 [cs.CL] 1 Jan 2024) Tj ET\n"
    data = _pdf(stamp + _line("The Real Title", 16, 72, 700) + _line("Body text.", 10, 72, 650))
    assert extract(data, "pdf").title == "The Real Title"


def test_pdf_title_keeps_small_capitals_on_the_title_line():
    # Small caps: full-size capitals, the rest of each word smaller, one baseline.
    line = ("BT /F1 17 Tf 72 700 Td (S) Tj /F1 13 Tf (ELF) Tj /F1 17 Tf (-RAG: L) Tj"
            " /F1 13 Tf (EARNING) Tj ET\n")
    data = _pdf(line + _line("Body text.", 10, 72, 650))
    assert extract(data, "pdf").title == "SELF-RAG: LEARNING"


def test_pdf_title_skips_a_drop_cap():
    # An IEEE-style drop cap is the largest character on the page.
    data = _pdf(_line("A Survey of Things", 18, 72, 700)
                + _line("L", 40, 72, 560) + _line("arge models are everywhere.", 10, 110, 590))
    assert extract(data, "pdf").title == "A Survey of Things"


def test_pdf_title_joins_a_wrapped_title_and_repairs_its_hyphen():
    data = _pdf(_line("Can Language Models Fol-", 17, 72, 700)
                + _line("low System Messages?", 17, 72, 680)
                + _line("Ada Fixtura", 12, 72, 650))
    assert extract(data, "pdf").title == "Can Language Models Follow System Messages?"


def test_pdf_title_is_the_first_block_of_its_largest_text():
    data = _pdf(_line("The Real Title", 17, 72, 700)
                + _line("Body text.", 10, 72, 650)
                + _line("Another Heading", 17, 72, 400))
    assert extract(data, "pdf").title == "The Real Title"


def test_pdf_with_no_text_on_its_first_page_has_no_title():
    assert extract(_pdf(""), "pdf").title is None


def test_pdf_title_rejects_interleaved_lines():
    # Two lines drawn over each other read back as alternating letters.
    data = _pdf(_line("Do the Laws of Physics", 17, 72, 700)
                + _line("Forbid Time Machines", 17, 76, 701))
    assert extract(data, "pdf").title is None


def test_pdf_title_keeps_two_large_capitals_on_a_small_caps_line():
    # Too few to set the title's size, but they sit on its line.
    line = ("BT /F1 17 Tf 72 700 Td (M) Tj /F1 13 Tf (EMORIZING ) Tj /F1 17 Tf (T) Tj"
            " /F1 13 Tf (RANSFORMERS) Tj ET\n")
    data = _pdf(line + _line("Body text.", 10, 72, 650))
    assert extract(data, "pdf").title == "MEMORIZING TRANSFORMERS"


def test_pdf_title_leaves_out_smaller_text_beside_it():
    # A journal's address in the margin, level with the title's top but smaller.
    data = _pdf(_line("A Commentary on Maxwell", 24, 72, 700)
                + _line("journal.example.org", 13, 400, 712))
    assert extract(data, "pdf").title == "A Commentary on Maxwell"


def test_pdf_title_rejects_a_page_number():
    data = _pdf(_line("[ 583 ]", 20, 72, 740) + _line("Body text of the first page.", 10, 72, 700))
    assert extract(data, "pdf").title is None


@pytest.mark.parametrize("raw, want", [
    ("(cid:204)RULER: What’s the Real Context Size", "RULER: What’s the Real Context Size"),
    ("(cid:69)(cid:113)(cid:117)(cid:105)", None),         # glyphs with no text mapping
    ("[ 583 ]", None),                                       # a page number
    ("AP21-3", None),                                        # a report code
    ("STRICTLY PRIVATE & CONFIDENTIAL", None),
    ("STRICTLYPRIVATE&CONFIDENTIAL", None),
    ("UNLIMITED DISTRIBUTION MASTER", None),
    ("Downloaded 23 Jul 2008. Redistribution subject to license; see http://example.org/terms",
     None),
    ("This is a list of projects. " * 8, None),             # a paragraph, not a title
    ("Confidential Computing for Machine Learning", "Confidential Computing for Machine Learning"),
    ("G-EVAL: NLG Evaluation using GPT-4", "G-EVAL: NLG Evaluation using GPT-4"),
    ("MTEB", "MTEB"),
])
def test_first_page_title_candidates_are_cleaned_or_rejected(raw, want):
    from alexandria.ingest.extract import _clean_pdf_title

    assert _clean_pdf_title(raw) == want

def test_markdown_title_is_its_first_h1():
    data = b"Some preamble.\n\n# The Real Title\n\n## A section\n\n# A later H1\n"
    assert extract(data, "md").title == "The Real Title"


def test_markdown_title_drops_emphasis_markers():
    assert extract(b"# **Bold Title**\n\nbody\n", "md").title == "Bold Title"


def test_markdown_title_skips_comments_in_code_fences():
    data = b"```sh\n# install the deps\npip install x\n```\n\n# Setup Notes\n"
    assert extract(data, "md").title == "Setup Notes"


def test_markdown_without_an_h1_has_no_title():
    assert extract(b"## Only a section\n\nbody\n", "md").title is None


def test_plain_text_hash_lines_are_not_titles():
    # In a .txt file a "# " line is as often a comment as a heading.
    assert extract(b"# generated by a script\nkey = value\n", "txt").title is None


def test_marker_title_is_its_first_h1_without_emphasis(monkeypatch):
    import sys
    import types

    from alexandria.ingest.extract import _extract_pdf_marker

    for name in ("marker", "marker.converters", "marker.converters.pdf",
                 "marker.models", "marker.output"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["marker.converters.pdf"].PdfConverter = type(
        "P", (), {"__init__": lambda self, *a, **k: None,
                  "__call__": lambda self, path: object()}
    )
    sys.modules["marker.models"].create_model_dict = lambda: {}
    sys.modules["marker.output"].text_from_rendered = (
        lambda r: ("Preprint\n\n# **A Marker Title**\n\nBody.", None, [])
    )
    monkeypatch.setenv("TORCH_DEVICE", "cpu")

    _, title, *_ = _extract_pdf_marker(b"%PDF-stub", device="cpu")
    assert title == "A Marker Title"
