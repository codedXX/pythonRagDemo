"""PDF parser built on PyMuPDF.

PyMuPDF is used rather than pypdf because Chinese multi-column layouts and CJK
font handling need a real layout engine; pypdf tends to produce reordered or
garbled text on the same files.

Two behaviours matter for correctness:

* Scanned PDFs. A page with no text layer yields nothing, and ingesting nothing
  would create a document that can never answer a question. Character density is
  checked instead and such files are rejected with an explicit reason.
* Repeated page furniture. Headers and footers repeat on every page and would
  consume chunk budget while contributing no information. Deciding what is
  furniture needs care: "appears on several pages" alone is not sufficient,
  because templated documents also repeat body prose, and stripping that silently
  deletes content. Furniture is therefore identified by repeating *and* sitting in
  the top or bottom margin band of the page.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf  # the modern import name; 'fitz' is deprecated

from rag_demo.parsers.base import (
    ParsedDocument,
    ScannedDocumentError,
    SegmentKind,
    StructuralSegment,
)

# A page needs at least this many extractable characters per page on average to
# be considered to have a usable text layer.
_SENTENCE_END = re.compile(r"[。！？.!?;；:：]\s*$")


@dataclass(slots=True)
class _Line:
    """One rendered line with its position, used for furniture detection."""

    text: str
    y_ratio: float


@dataclass(slots=True)
class PdfParser:
    """Turn a PDF into structural segments, one per paragraph."""

    min_chars_per_page: int = 40
    # A line repeating on at least this fraction of pages is page furniture.
    furniture_page_fraction: float = 0.4
    furniture_min_pages: int = 3
    # Only lines inside this fraction of the page height at the top or bottom are
    # eligible. The value is bounded by real page geometry: on A4, body text
    # typically begins around y=72 of 842, i.e. a ratio of 0.086, so a band of 0.08
    # keeps the first body line while still catching a header at y≈30 (ratio 0.036).
    # A wider band such as 0.12 would classify the first body line of every normal
    # document as furniture.
    furniture_margin_ratio: float = 0.08

    def parse(self, name: str, content: bytes) -> ParsedDocument:
        try:
            document = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"cannot open PDF: {exc}") from exc

        try:
            page_count = document.page_count
            page_texts: list[str] = []
            page_lines: list[list[_Line]] = []
            for index in range(page_count):
                page = document.load_page(index)
                page_texts.append(page.get_text("text"))
                page_lines.append(self._lines_of(page))
        finally:
            document.close()

        total_chars = sum(len(t.strip()) for t in page_texts)
        if page_count > 0:
            density = total_chars / page_count
            if density < self.min_chars_per_page:
                raise ScannedDocumentError(
                    f"PDF appears to be image-based: {total_chars} extractable "
                    f"characters across {page_count} page(s), which is "
                    f"{density:.0f} per page (minimum {self.min_chars_per_page}). "
                    "Such a file would need OCR, which is not supported."
                )

        furniture = self._detect_furniture(page_lines)
        segments = self._segments_from(page_texts, furniture)

        # Safety net: if stripping furniture left nothing, the detector was wrong
        # about this document. Keep the content rather than returning an empty
        # document that would fail mysteriously later.
        if not segments and furniture:
            segments = self._segments_from(page_texts, set())
            furniture = set()

        return ParsedDocument(
            name=name,
            segments=segments,
            page_count=page_count,
            char_count=total_chars,
            metadata={"format": "pdf", "removed_furniture": len(furniture)},
        )

    # ------------------------------------------------------------------ parsing

    def _lines_of(self, page: "pymupdf.Page") -> list[_Line]:
        """Extract lines with their vertical position relative to page height."""
        height = page.rect.height or 1.0
        lines: list[_Line] = []
        try:
            payload = page.get_text("dict")
        except Exception:
            return lines
        for block in payload.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not text.strip():
                    continue
                y0 = line.get("bbox", [0, 0, 0, 0])[1]
                lines.append(_Line(text=text.strip(), y_ratio=y0 / height))
        return lines

    def _detect_furniture(self, page_lines: list[list[_Line]]) -> set[str]:
        """Find lines that repeat across pages within the margin bands."""
        if len(page_lines) < self.furniture_min_pages:
            return set()

        counts: Counter[str] = Counter()
        for lines in page_lines:
            in_margin = {
                line.text
                for line in lines
                if line.y_ratio <= self.furniture_margin_ratio
                or line.y_ratio >= 1.0 - self.furniture_margin_ratio
            }
            # Counted once per page so a line repeated within a page is not
            # mistaken for a running header.
            counts.update(in_margin)

        threshold = max(
            self.furniture_min_pages,
            int(len(page_lines) * self.furniture_page_fraction),
        )
        return {text for text, count in counts.items() if count >= threshold}

    def _segments_from(
        self, page_texts: list[str], furniture: set[str]
    ) -> list[StructuralSegment]:
        segments: list[StructuralSegment] = []
        for index, page_text in enumerate(page_texts, start=1):
            for paragraph in self._paragraphs(page_text, furniture):
                segments.append(
                    StructuralSegment(
                        text=paragraph,
                        kind=SegmentKind.PARAGRAPH,
                        page=index,
                        heading_path=(),
                    )
                )
        return segments

    def _paragraphs(self, page_text: str, furniture: set[str]) -> list[str]:
        """Group page lines into paragraphs, dropping furniture and page numbers."""
        paragraphs: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                joined = _join_lines(buffer)
                if joined.strip():
                    paragraphs.append(joined.strip())
                buffer.clear()

        for raw_line in page_text.split("\n"):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if line in furniture:
                continue
            if _is_page_number(line):
                continue
            buffer.append(line)
            # A sentence-terminated line followed by substantial text is a
            # plausible paragraph boundary even without blank lines, which is
            # common in PDFs exported from word processors.
            if _SENTENCE_END.search(line) and len(" ".join(buffer)) > 400:
                flush()

        flush()
        return paragraphs


def _is_page_number(line: str) -> bool:
    """Whether a line is just a page number, possibly decorated."""
    stripped = line.strip()
    if not stripped or len(stripped) > 20:
        return False
    return bool(re.fullmatch(r"[-–—\s]*(?:\d{1,4}|[ivxlcIVXLC]{1,6})[-–—\s]*", stripped))


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines, inserting a space only where Latin text needs one.

    Chinese text wraps without spaces, so unconditionally joining with a space
    would corrupt it; unconditionally joining without one would run English words
    together.
    """
    if not lines:
        return ""
    result = lines[0]
    for line in lines[1:]:
        if _needs_space(result, line):
            result += " " + line
        else:
            result += line
    return result


def _needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    last = left[-1]
    first = right[0]
    # A hyphen at a line break continues the word.
    if last == "-":
        return False
    ascii_left = last.isascii() and (last.isalnum() or last in ",.;:!?)")
    ascii_right = first.isascii() and first.isalnum()
    return ascii_left and ascii_right