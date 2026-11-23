"""Markdown parser.

Markdown carries explicit structure, so it is parsed structurally rather than by
scanning for blank lines. Heading levels build a breadcrumb path that every
segment inherits, and fenced code blocks are kept whole so they are never split
mid-statement.

Chinese Markdown in particular often has no blank line between a heading and the
first paragraph, so paragraph accumulation is driven by block boundaries instead
of blank-line runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_demo.parsers.base import (
    ParsedDocument,
    SegmentKind,
    StructuralSegment,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_QUOTE = re.compile(r"^\s*>\s?")
_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$")


@dataclass(slots=True)
class MarkdownParser:
    """Turn Markdown source into structural segments."""

    def parse(self, name: str, content: bytes) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace")
        segments = self.parse_text(text)
        return ParsedDocument(
            name=name,
            segments=segments,
            page_count=1,
            char_count=sum(len(s.text) for s in segments),
            metadata={"format": "markdown"},
        )

    def parse_text(self, text: str) -> list[StructuralSegment]:
        """Parse Markdown text into segments, tracking the heading path."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        segments: list[StructuralSegment] = []
        # One entry per heading level; index 0 is the level-1 heading.
        heading_stack: list[str] = []
        buffer: list[str] = []
        buffer_kind = SegmentKind.PARAGRAPH
        in_fence = False
        fence_marker = ""

        def flush() -> None:
            nonlocal buffer, buffer_kind
            body = "\n".join(buffer).strip("\n")
            if body.strip():
                segments.append(
                    StructuralSegment(
                        text=body,
                        kind=buffer_kind,
                        depth=0,
                        heading_path=tuple(heading_stack),
                    )
                )
            buffer = []
            buffer_kind = SegmentKind.PARAGRAPH

        for raw_line in lines:
            # Fenced code blocks are consumed verbatim, including blank lines and
            # anything that would otherwise look like a heading.
            fence_match = _FENCE.match(raw_line)
            if in_fence:
                buffer.append(raw_line)
                if fence_match and raw_line.strip().startswith(fence_marker):
                    in_fence = False
                    flush()
                continue
            if fence_match:
                flush()
                in_fence = True
                fence_marker = fence_match.group(1)
                buffer_kind = SegmentKind.CODE
                buffer = [raw_line]
                continue

            heading = _HEADING.match(raw_line)
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                # Truncate the path to the parent level, then push this heading.
                heading_stack = heading_stack[: level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")
                heading_stack.append(title)
                segments.append(
                    StructuralSegment(
                        text=raw_line.strip(),
                        kind=SegmentKind.HEADING,
                        depth=level,
                        heading_path=tuple(h for h in heading_stack if h),
                    )
                )
                continue

            if _HR.match(raw_line):
                flush()
                continue

            if not raw_line.strip():
                flush()
                continue

            if _QUOTE.match(raw_line):
                if buffer and buffer_kind is not SegmentKind.QUOTE:
                    flush()
                buffer_kind = SegmentKind.QUOTE
                buffer.append(_QUOTE.sub("", raw_line))
                continue

            if _TABLE_ROW.match(raw_line):
                if buffer and buffer_kind is not SegmentKind.TABLE:
                    flush()
                buffer_kind = SegmentKind.TABLE
                buffer.append(raw_line)
                continue

            if _LIST_ITEM.match(raw_line):
                if buffer and buffer_kind is not SegmentKind.LIST:
                    flush()
                buffer_kind = SegmentKind.LIST
                buffer.append(raw_line)
                continue

            # Plain text: joins a list or paragraph depending on what is open.
            if buffer_kind is SegmentKind.CODE:
                buffer_kind = SegmentKind.PARAGRAPH
            buffer.append(raw_line)

        flush()
        return segments