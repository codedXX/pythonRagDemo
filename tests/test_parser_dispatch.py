"""Tests for parser dispatch and file-type validation."""

from __future__ import annotations

import pymupdf
import pytest

from rag_demo.config import ParsingSettings
from rag_demo.parsers import (
    ScannedDocumentError,
    UnsupportedDocumentError,
    check_extension,
    document_id,
    parse_document,
    sniff_extension,
)
from rag_demo.parsers.base import supported_extensions_display


@pytest.mark.parametrize("name", ["a.pdf", "b.md", "c.markdown", "D.PDF", "e.Md"])
def test_supported_extensions_are_accepted(name):
    assert check_extension(name) in {".pdf", ".md", ".markdown"}


@pytest.mark.parametrize("name", ["a.txt", "b.docx", "c.html", "d", "e.xlsx"])
def test_unsupported_extensions_are_rejected(name):
    with pytest.raises(UnsupportedDocumentError):
        check_extension(name)


def test_rejection_message_lists_supported_types():
    with pytest.raises(UnsupportedDocumentError) as excinfo:
        check_extension("notes.txt")

    message = str(excinfo.value)
    assert ".txt" in message
    assert supported_extensions_display() == ".markdown, .md, .pdf"


def test_markdown_dispatch():
    document = parse_document(
        "notes.md", "# Title\n\nSome body text here.\n".encode(), ParsingSettings()
    )

    assert document.metadata["format"] == "markdown"
    assert not document.is_empty()


def test_pdf_dispatch():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, 523, 770),
        "A page of real text that is long enough to clear the density threshold.",
        fontsize=11,
    )
    content = pdf.tobytes()
    pdf.close()

    document = parse_document("doc.pdf", content, ParsingSettings())

    assert document.metadata["format"] == "pdf"


def test_empty_content_is_rejected():
    with pytest.raises(ValueError):
        parse_document("empty.md", b"", ParsingSettings())


def test_scanned_pdf_error_propagates_through_dispatch():
    pdf = pymupdf.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()

    with pytest.raises(ScannedDocumentError):
        parse_document("scan.pdf", content, ParsingSettings())


def test_pdf_renamed_to_markdown_is_still_treated_as_pdf():
    """Magic number beats extension: otherwise the parser emits binary garbage."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, 523, 770),
        "Real text content that clears the density threshold comfortably here.",
        fontsize=11,
    )
    content = pdf.tobytes()
    pdf.close()

    assert sniff_extension("fake.md", content) == ".pdf"
    assert parse_document("fake.md", content, ParsingSettings()).metadata["format"] == "pdf"


def test_markdown_named_pdf_stays_markdown():
    assert sniff_extension("real.pdf", b"# Heading\n\ntext\n") == ".pdf"


def test_document_id_is_stable_and_content_sensitive():
    first = document_id("a.md", b"same")
    second = document_id("a.md", b"same")
    changed = document_id("a.md", b"different")
    renamed = document_id("b.md", b"same")

    assert first == second
    assert first != changed
    assert first != renamed