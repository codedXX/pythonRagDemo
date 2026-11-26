"""用 markdown-it-py 识别顶层块，保留原文及其所属章节。"""

from markdown_it import MarkdownIt

from rag_demo.parsers.base import ParsedDocument, SegmentKind, StructuralSegment


# 让链接定义也带上源行范围，保留文档中可用于检索的链接目标。
_PARSER = MarkdownIt("commonmark", {"inline_definitions": True}).enable("table")
_BLOCK_KINDS = {
    "heading_open": SegmentKind.HEADING,
    "paragraph_open": SegmentKind.PARAGRAPH,
    "html_block": SegmentKind.PARAGRAPH,
    "definition": SegmentKind.PARAGRAPH,
    "bullet_list_open": SegmentKind.LIST,
    "ordered_list_open": SegmentKind.LIST,
    "blockquote_open": SegmentKind.QUOTE,
    "table_open": SegmentKind.TABLE,
    "fence": SegmentKind.CODE,
    "code_block": SegmentKind.CODE,
}


def parse_markdown(name: str, content: bytes) -> ParsedDocument:
    text = content.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    tokens = _PARSER.parse(text)
    headings: list[tuple[int, str]] = []
    segments: list[StructuralSegment] = []

    for index, token in enumerate(tokens):
        kind = _BLOCK_KINDS.get(token.type)
        if token.level != 0 or kind is None:
            continue

        depth = 0
        if kind is SegmentKind.HEADING:
            depth = int(token.tag[1:])
            while headings and headings[-1][0] >= depth:
                headings.pop()
            # heading_open 后的 inline 节点保存去除标题标记的文字。
            title = tokens[index + 1].content.strip()
            if title:
                headings.append((depth, title))

        start, end = token.map
        segments.append(
            StructuralSegment(
                text="\n".join(lines[start:end]),
                kind=kind,
                depth=depth,
                heading_path=tuple(title for _, title in headings),
            )
        )

    return ParsedDocument(
        name=name,
        segments=segments,
        page_count=1,
        char_count=sum(len(segment.text) for segment in segments),
        metadata={"format": "markdown"},
    )
