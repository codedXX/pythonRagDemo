"""按 Markdown 块结构解析文档。各片段继承所属标题路径，围栏代码整体保留。段落随块边界结束，因此标题和中文正文之间没有空行也能正确识别。"""

from __future__ import annotations

import re

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


def parse_markdown(name: str, content: bytes) -> ParsedDocument:
    text = content.decode("utf-8", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    segments: list[StructuralSegment] = []
    # 每级标题占一个位置，下标零对应一级标题。
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
        # 逐行原样保留围栏内的内容，包括空行和形似标题的行。
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
            # 保留上级标题，再将当前标题加入路径。
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

        # 根据当前打开的块，将普通文本并入列表或段落。
        if buffer_kind is SegmentKind.CODE:
            buffer_kind = SegmentKind.PARAGRAPH
        buffer.append(raw_line)

    flush()
    return ParsedDocument(
        name=name,
        segments=segments,
        page_count=1,
        char_count=sum(len(s.text) for s in segments),
        metadata={"format": "markdown"},
    )
