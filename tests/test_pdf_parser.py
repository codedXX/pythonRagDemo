"""Tests for the PDF parser, using PDFs generated at test time.

Generating real PDFs (rather than mocking PyMuPDF) means these tests exercise the
same code path as production, and the scanned-file rejection is verified against an
actual text-free page.
"""

from __future__ import annotations

import pymupdf
import pytest

from rag_demo.parsers.base import ScannedDocumentError
from rag_demo.parsers.pdf import PdfParser, _is_page_number, _needs_space


def _build_pdf(pages: list[str], header: str | None = None) -> bytes:
    """Create a PDF with one text block per supplied page.

    ``header`` is drawn at the very top of each page, inside the margin band where
    running furniture actually lives.
    """
    document = pymupdf.open()
    for body in pages:
        page = document.new_page()
        # insert_textbox wraps text; insert_text does not. Wrapping matters
        # because the parser's paragraph joining is what is under test.
        page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), body, fontsize=11)
        if header:
            page.insert_text((72, 30), header, fontsize=9)
    data = document.tobytes()
    document.close()
    return data


def _blank_pdf(pages: int = 2) -> bytes:
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()  # no text layer at all, like a scan
    data = document.tobytes()
    document.close()
    return data


def test_parses_pages_and_records_page_numbers():
    data = _build_pdf(
        [
            "First page sentence number one, written long enough to be a real page of text.",
            "Second page content, also long enough that the file is not mistaken for a scan.",
        ]
    )

    document = PdfParser().parse("sample.pdf", data)

    assert document.page_count == 2
    assert document.metadata["format"] == "pdf"
    pages_seen = {segment.page for segment in document.segments}
    assert pages_seen == {1, 2}


def test_text_is_extracted_in_readable_form():
    data = _build_pdf(["The quick brown fox jumps over the lazy dog. " * 3])

    document = PdfParser().parse("sample.pdf", data)
    joined = " ".join(s.text for s in document.segments)

    assert "quick brown fox" in joined


def test_scanned_pdf_is_rejected():
    """A PDF with no text layer must be refused, not ingested as empty content."""
    with pytest.raises(ScannedDocumentError) as excinfo:
        PdfParser().parse("scan.pdf", _blank_pdf())

    message = str(excinfo.value)
    assert "image-based" in message
    assert "OCR" in message


def test_low_density_pdf_is_rejected():
    # One short line on each of many pages: technically has text, effectively a
    # scan plus a stamp.
    data = _build_pdf(["x"] * 10)

    with pytest.raises(ScannedDocumentError):
        PdfParser(min_chars_per_page=40).parse("sparse.pdf", data)


def test_density_threshold_is_configurable():
    data = _build_pdf(["A short but real sentence of text, repeated to reach a plausible page length. " * 2] * 3)

    # With an impossibly high threshold the file is rejected...
    with pytest.raises(ScannedDocumentError):
        PdfParser(min_chars_per_page=10_000).parse("sample.pdf", data)

    # ...and with a low one it passes.
    assert not PdfParser(min_chars_per_page=1).parse("sample.pdf", data).is_empty()


def test_repeated_header_is_removed():
    """Running headers repeat in the top margin and should not consume budget."""
    pages = [f"Unique body text for page {i} with enough characters to matter." for i in range(5)]

    document = PdfParser().parse("report.pdf", _build_pdf(pages, header="Running Header"))

    joined = " ".join(s.text for s in document.segments)
    assert "Running Header" not in joined
    assert "Unique body text" in joined


def test_body_text_repeating_on_every_page_is_kept():
    """Regression: templated documents repeat body prose on every page.

    An earlier rule treated any line repeating across pages as furniture, which
    deleted the whole body of such files. Position, not repetition alone, decides.
    """
    repeated = "This exact sentence appears on every page because the document is a template. "
    pages = [repeated * 2 + f"Page specific value {i}." for i in range(5)]

    document = PdfParser().parse("template.pdf", _build_pdf(pages))

    assert any("appears on every page" in s.text for s in document.segments)


def test_page_numbers_are_dropped():
    """A bare page number line is noise and is filtered before chunking."""
    body = "Real content that should survive the filter, long enough to look like a page. " * 3
    data = _build_pdf([body + "\n2", body + "\n3", body + "\n4"])

    document = PdfParser().parse("doc.pdf", data)

    assert not any(s.text.strip().isdigit() for s in document.segments)


def test_corrupt_pdf_raises_value_error():
    with pytest.raises(ValueError):
        PdfParser().parse("broken.pdf", b"not a pdf at all")


def test_removed_furniture_is_counted():
    pages = [f"Body {i} with distinct wording that fills the page out properly." for i in range(4)]

    document = PdfParser().parse("doc.pdf", _build_pdf(pages, header="Same Header"))

    assert document.metadata["removed_furniture"] >= 1


class TestHelpers:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("12", True),
            ("- 12 -", True),
            ("iv", True),
            ("Page 12 of 30", False),
            ("2024", True),
            ("a1", False),
            ("", False),
        ],
    )
    def test_page_number_detection(self, line, expected):
        assert _is_page_number(line) is expected

    def test_space_inserted_between_latin_words(self):
        assert _needs_space("hello", "world") is True

    def test_no_space_inserted_for_chinese(self):
        # Chinese wraps without spaces; inserting one would corrupt the text.
        assert _needs_space("这是第一句。", "这是第二句。") is False

    def test_no_space_after_hyphenated_break(self):
        assert _needs_space("inter-", "national") is False


def test_document_is_never_emptied_by_furniture_detection():
    """Whatever the detector concludes, a file with text must yield segments."""
    pages = ["Header Text"] * 6

    document = PdfParser(min_chars_per_page=1).parse("headers.pdf", _build_pdf(pages))

    assert not document.is_empty()
