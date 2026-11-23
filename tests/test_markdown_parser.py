"""Tests for the Markdown parser."""

from __future__ import annotations

from rag_demo.parsers.base import SegmentKind
from rag_demo.parsers.markdown import MarkdownParser


def _parse(text: str):
    return MarkdownParser().parse_text(text)


def test_heading_path_is_tracked_through_nesting():
    segments = _parse(
        "# Top\n\nintro\n\n## Section A\n\nbody a\n\n### Detail A1\n\ndeep text\n"
    )

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]
    assert paragraphs[0].heading_path == ("Top",)
    assert paragraphs[1].heading_path == ("Top", "Section A")
    # The third-level heading extends, not replaces, the path.
    assert paragraphs[2].heading_path == ("Top", "Section A", "Detail A1")


def test_heading_path_resets_on_sibling_heading():
    segments = _parse("# Top\n\n## A\n\ntext a\n\n## B\n\ntext b\n")

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]
    assert paragraphs[1].heading_path == ("Top", "B")


def test_heading_path_returns_to_parent_after_subsection():
    segments = _parse("# Top\n\n## A\n\n### Deep\n\ndeep\n\n## B\n\nshallow\n")

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]
    assert paragraphs[-1].heading_path == ("Top", "B")


def test_headings_are_emitted_as_segments():
    segments = _parse("# Title\n\nbody\n")

    headings = [s for s in segments if s.kind is SegmentKind.HEADING]
    assert [h.depth for h in headings] == [1]
    assert "Title" in headings[0].text


def test_code_fence_is_kept_whole():
    source = "# Code\n\n```python\n# this looks like a heading\nprint('hi')\n\nx = 1\n```\n\nafter\n"

    segments = _parse(source)

    code = [s for s in segments if s.kind is SegmentKind.CODE]
    assert len(code) == 1
    # The '#' inside the fence must not have become a heading, and the blank line
    # inside it must not have split the block.
    assert "# this looks like a heading" in code[0].text
    assert "x = 1" in code[0].text
    assert not any(s.kind is SegmentKind.HEADING and "looks like" in s.text for s in segments)


def test_unclosed_fence_does_not_lose_text():
    segments = _parse("```\nunterminated code\nstill here\n")

    code = [s for s in segments if s.kind is SegmentKind.CODE]
    assert code and "still here" in code[0].text


def test_list_items_form_one_segment():
    segments = _parse("- one\n- two\n- three\n")

    lists = [s for s in segments if s.kind is SegmentKind.LIST]
    assert len(lists) == 1
    assert lists[0].text.count("-") >= 3


def test_ordered_list_is_recognised():
    segments = _parse("1. first\n2. second\n")

    assert [s.kind for s in segments] == [SegmentKind.LIST]


def test_table_rows_group_together():
    segments = _parse("| a | b |\n| --- | --- |\n| 1 | 2 |\n")

    tables = [s for s in segments if s.kind is SegmentKind.TABLE]
    assert len(tables) == 1


def test_blockquote_is_separated_from_paragraph():
    segments = _parse("normal text\n\n> quoted line\n> continues\n")

    assert any(s.kind is SegmentKind.QUOTE for s in segments)
    assert segments[0].kind is SegmentKind.PARAGRAPH


def test_horizontal_rule_separates_paragraphs():
    segments = _parse("above\n\n---\n\nbelow\n")

    texts = [s.text for s in segments if s.kind is SegmentKind.PARAGRAPH]
    assert texts == ["above", "below"]


def test_chinese_text_without_blank_lines_is_preserved():
    """Chinese Markdown often omits blank lines; lines must still accumulate."""
    segments = _parse("## 标题\n这是第一句。\n这是第二句。\n")

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]
    assert len(paragraphs) == 1
    assert "第一句" in paragraphs[0].text and "第二句" in paragraphs[0].text


def test_breadcrumb_property_joins_path():
    segments = _parse("# A\n\n## B\n\ntext\n")

    paragraph = [s for s in segments if s.kind is SegmentKind.PARAGRAPH][0]
    assert paragraph.breadcrumb == "A > B"


def test_parse_handles_bytes_and_reports_metadata():
    document = MarkdownParser().parse("notes.md", "# T\n\nbody\n".encode())

    assert document.name == "notes.md"
    assert document.page_count == 1
    assert document.char_count > 0
    assert document.metadata["format"] == "markdown"
    assert not document.is_empty()


def test_empty_document_is_reported_empty():
    assert MarkdownParser().parse("empty.md", b"").is_empty()