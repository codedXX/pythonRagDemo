"""Shared document model produced by the parsers.

Parsers turn bytes into a flat sequence of :class:`StructuralSegment` values plus
a provenance header. Keeping the intermediate representation this simple makes the
two parsers interchangeable and lets the chunker stay ignorant of file formats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SegmentKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    QUOTE = "quote"


@dataclass(slots=True)
class StructuralSegment:
    """A structurally coherent run of text, with its provenance.

    ``depth`` is the heading level for headings and 0 elsewhere. ``heading_path``
    records the enclosing headings at the point the segment started, which becomes
    the breadcrumb shown in citations and supplied to the reranker.
    """

    text: str
    kind: SegmentKind
    page: int = 0
    depth: int = 0
    heading_path: tuple[str, ...] = ()

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)


@dataclass(slots=True)
class ParsedDocument:
    """Everything a parser knows about a document."""

    name: str
    segments: list[StructuralSegment]
    page_count: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(segment.text.strip() for segment in self.segments)

    def chars_per_page(self) -> float:
        if self.page_count <= 0:
            return float("inf") if self.char_count == 0 else float(self.char_count)
        return self.char_count / self.page_count


class UnsupportedDocumentError(ValueError):
    """The file type is not one we ingest."""


class ScannedDocumentError(ValueError):
    """The document has no usable text layer and would require OCR."""


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown"})


def supported_extensions_display() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def check_extension(name: str) -> str:
    """Validate the extension and return it lower-cased."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        shown = suffix or "(no extension)"
        raise UnsupportedDocumentError(
            f"unsupported file type {shown}; supported types are "
            f"{supported_extensions_display()}"
        )
    return suffix


def document_id(name: str, content: bytes) -> str:
    """Stable identifier for a document.

    Derived from the file name and its bytes so re-uploading the same file is
    recognised as the same document rather than creating a duplicate, while a
    changed file gets a new identity.
    """
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(content)
    return digest.hexdigest()[:32]