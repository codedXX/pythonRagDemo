"""用 PyMuPDF 从 PDF 版面提取文本，以保留中文等多字节文字和页面位置信息。

按每页平均可提取字符数识别缺少可用文本层的扫描件。只有跨页重复且位于页边距内的文字才视为页眉或页脚，避免误删模板文档中重复的正文。"""

from __future__ import annotations

import re
from collections import Counter

import pymupdf

from rag_demo.parsers.base import (
    ParsedDocument,
    ScannedDocumentError,
    SegmentKind,
    StructuralSegment,
)

MARGIN_RATIO = 0.08
_SENTENCE_END = re.compile(r"[。！？.!?;；:：]\s*$")


def parse_pdf(name: str, content: bytes, min_chars_per_page: int = 40) -> ParsedDocument:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"cannot open PDF: {exc}") from exc

    with document:
        page_count = document.page_count
        page_texts: list[str] = []
        margin_counts: Counter[str] = Counter()
        for page in document:
            page_texts.append(page.get_text("text"))
            if page_count >= 3:
                margin_counts.update(_margin_texts(page))

    total_chars = sum(len(t.strip()) for t in page_texts)
    if page_count > 0:
        density = total_chars / page_count
        if density < min_chars_per_page:
            raise ScannedDocumentError(
                f"PDF appears to be image-based: {total_chars} extractable "
                f"characters across {page_count} page(s), which is "
                f"{density:.0f} per page (minimum {min_chars_per_page}). "
                "Such a file would need OCR, which is not supported."
            )

    # 一段边缘文字至少出现在三页，才可能是重复页眉或页脚。
    threshold = max(3, int(page_count * 0.4))
    furniture = {text for text, count in margin_counts.items() if count >= threshold}
    segments = _segments_from(page_texts, furniture)

    # 如果过滤后没有正文，则保留原内容，避免误判导致整份文档为空。
    if not segments and furniture:
        segments = _segments_from(page_texts, set())
        furniture = set()

    return ParsedDocument(
        name=name,
        segments=segments,
        page_count=page_count,
        char_count=total_chars,
        metadata={"format": "pdf", "removed_furniture": len(furniture)},
    )


def _margin_texts(page: pymupdf.Page) -> set[str]:
    """每页只统计一次重复文字；缺少版面信息时仍可以使用纯文本。"""
    try:
        layout = page.get_text("dict")
    except Exception:
        return set()

    texts = set()
    height = page.rect.height or 1.0
    for block in layout["blocks"]:
        for line in block.get("lines", []):
            y = line["bbox"][1] / height
            if y <= MARGIN_RATIO or y >= 1.0 - MARGIN_RATIO:
                text = "".join(span["text"] for span in line["spans"]).strip()
                if text:
                    texts.add(text)
    return texts


def _segments_from(page_texts: list[str], furniture: set[str]) -> list[StructuralSegment]:
    """在空行、较长句末或页面末尾结段，保留每段的页码。"""
    segments = []
    for page_number, text in enumerate(page_texts, start=1):
        lines = []
        for raw in (text + "\n").split("\n"):
            line = raw.strip()
            if line:
                if line in furniture or _is_page_number(line):
                    continue
                lines.append(line)
                if not (_SENTENCE_END.search(line) and len(" ".join(lines)) > 400):
                    continue
            if lines:
                segments.append(StructuralSegment(
                    text=_join_lines(lines), kind=SegmentKind.PARAGRAPH, page=page_number
                ))
                lines.clear()
    return segments


def _is_page_number(line: str) -> bool:
    """识别单独成行的页码以及带装饰符的页码。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 20:
        return False
    return bool(re.fullmatch(r"[-–—\s]*(?:\d{1,4}|[ivxlcIVXLC]{1,6})[-–—\s]*", stripped))


def _join_lines(lines: list[str]) -> str:
    """拼接换行文本；中文直接连接，拉丁文字之间按需补空格，避免单词粘连。"""
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
    # 行末连字符表示单词延续到下一行。
    if last == "-":
        return False
    ascii_left = last.isascii() and (last.isalnum() or last in ",.;:!?)")
    ascii_right = first.isascii() and first.isalnum()
    return ascii_left and ascii_right
