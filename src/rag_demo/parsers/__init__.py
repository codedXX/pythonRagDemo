"""Format dispatch for the parsers."""

from __future__ import annotations

from rag_demo.config import ParsingSettings
from rag_demo.parsers.base import (
    ParsedDocument,
    ScannedDocumentError,
    UnsupportedDocumentError,
    check_extension,
    document_id,
)
from rag_demo.parsers.markdown import MarkdownParser
from rag_demo.parsers.pdf import PdfParser

__all__ = [
    "ParsedDocument",
    "ScannedDocumentError",
    "UnsupportedDocumentError",
    "check_extension",
    "document_id",
    "parse_document",
]


def parse_document(name: str, content: bytes, settings: ParsingSettings) -> ParsedDocument:
    """Parse ``content`` according to the file extension of ``name``.

    Raises :class:`UnsupportedDocumentError` for a type we do not ingest and
    :class:`ScannedDocumentError` for a PDF without a usable text layer.
    """
    suffix = sniff_extension(name, content)

    if not content:
        raise ValueError(f"{name} is empty")

    if suffix == ".pdf":
        return PdfParser(min_chars_per_page=settings.min_chars_per_page).parse(name, content)
    return MarkdownParser().parse(name, content)


def sniff_extension(name: str, content: bytes) -> str:
    """Report the extension to trust for ``name``.

    A PDF renamed to ``.md`` would parse as garbled text, so the magic number wins
    when it contradicts the extension.
    """
    declared = check_extension(name)
    if content.startswith(b"%PDF-") and declared != ".pdf":
        return ".pdf"
    return declared