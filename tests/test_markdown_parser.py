"""验证 Markdown 的结构解析和标题路径。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

from rag_demo.parsers.base import SegmentKind  # 从 rag_demo.parsers.base 导入片段类型。
from rag_demo.parsers.markdown import parse_markdown  # 从 rag_demo.parsers.markdown 导入标记文本解析器。


def _parse(text: str):  # 定义解析响应的处理逻辑。
    return parse_markdown("test.md", text.encode()).segments  # 返回parse_text的结果。


def test_heading_path_is_tracked_through_nesting():  # 验证嵌套标题能正确扩展路径。
    segments = _parse(  # 将结构片段列表设为解析响应的结果。
        "# Top\n\nintro\n\n## Section A\n\nbody a\n\n### Detail A1\n\ndeep text\n"  # 构造三级嵌套标题并在各级放入正文。
    )  # 结束当前多行表达式。

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]  # 将段落列表设为从结构片段列表中提取的各项。
    assert paragraphs[0].heading_path == ("Top",)  # 确认标题路径等于组合值。
    assert paragraphs[1].heading_path == ("Top", "Section A")  # 确认标题路径等于组合值。
    # 三级标题应接在上级标题路径之后。
    assert paragraphs[2].heading_path == ("Top", "Section A", "Detail A1")  # 确认标题路径等于组合值。


def test_heading_path_resets_on_sibling_heading():  # 验证同级标题替换路径末端。
    segments = _parse("# Top\n\n## A\n\ntext a\n\n## B\n\ntext b\n")  # 将结构片段列表设为解析响应的结果。

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]  # 将段落列表设为从结构片段列表中提取的各项。
    assert paragraphs[1].heading_path == ("Top", "B")  # 确认标题路径等于组合值。


def test_heading_path_returns_to_parent_after_subsection():  # 验证离开子节后恢复上级路径。
    segments = _parse("# Top\n\n## A\n\n### Deep\n\ndeep\n\n## B\n\nshallow\n")  # 将结构片段列表设为解析响应的结果。

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]  # 将段落列表设为从结构片段列表中提取的各项。
    assert paragraphs[-1].heading_path == ("Top", "B")  # 确认标题路径等于组合值。


def test_headings_are_emitted_as_segments():  # 验证标题本身也产生结构片段。
    segments = _parse("# Title\n\nbody\n")  # 将结构片段列表设为解析响应的结果。

    headings = [s for s in segments if s.kind is SegmentKind.HEADING]  # 将headings设为从结构片段列表中提取的各项。
    assert [h.depth for h in headings] == [1]  # 确认从headings中提取的各项等于列表。
    assert "Title" in headings[0].text  # 确认文本包含“Title”。


def test_code_fence_is_kept_whole():  # 验证围栏代码保持完整。
    source = "# Code\n\n```python\n# this looks like a heading\nprint('hi')\n\nx = 1\n```\n\nafter\n"  # 设置来源。

    segments = _parse(source)  # 将结构片段列表设为解析响应的结果。

    code = [s for s in segments if s.kind is SegmentKind.CODE]  # 将编码设为从结构片段列表中提取的各项。
    assert len(code) == 1  # 确认编码的数量等于1。
    # 围栏中的井号是代码内容，空行也应保留在同一代码块中。
    assert "# this looks like a heading" in code[0].text  # 确认文本包含文本内容。
    assert "x = 1" in code[0].text  # 确认文本包含“x = 1”。
    assert not any(s.kind is SegmentKind.HEADING and "looks like" in s.text for s in segments)  # 确认不满足“是否存在满足条件的项的结果”。


def test_unclosed_fence_does_not_lose_text():  # 验证缺少结束围栏时保留已有代码。
    segments = _parse("```\nunterminated code\nstill here\n")  # 将结构片段列表设为解析响应的结果。

    code = [s for s in segments if s.kind is SegmentKind.CODE]  # 将编码设为从结构片段列表中提取的各项。
    assert code and "still here" in code[0].text  # 确认编码且“still here”包含在文本。


def test_list_items_form_one_segment():  # 验证连续列表项合并到同一片段。
    segments = _parse("- one\n- two\n- three\n")  # 将结构片段列表设为解析响应的结果。

    lists = [s for s in segments if s.kind is SegmentKind.LIST]  # 将lists设为从结构片段列表中提取的各项。
    assert len(lists) == 1  # 确认lists的数量等于1。
    assert lists[0].text.count("-") >= 3  # 确认数量的结果不小于3。


def test_ordered_list_is_recognised():  # 验证识别有序列表。
    segments = _parse("1. first\n2. second\n")  # 将结构片段列表设为解析响应的结果。

    assert [s.kind for s in segments] == [SegmentKind.LIST]  # 确认从结构片段列表中提取的各项等于SegmentKind.LIST。


def test_table_rows_group_together():  # 验证连续表格行归为同一片段。
    segments = _parse("| a | b |\n| --- | --- |\n| 1 | 2 |\n")  # 将结构片段列表设为解析响应的结果。

    tables = [s for s in segments if s.kind is SegmentKind.TABLE]  # 将tables设为从结构片段列表中提取的各项。
    assert len(tables) == 1  # 确认tables的数量等于1。


def test_blockquote_is_separated_from_paragraph():  # 验证引文与普通段落分开。
    segments = _parse("normal text\n\n> quoted line\n> continues\n")  # 将结构片段列表设为解析响应的结果。

    assert any(s.kind is SegmentKind.QUOTE for s in segments)  # 确认至少一项满足“片段的类别为SegmentKind.QUOTE”。
    assert segments[0].kind is SegmentKind.PARAGRAPH  # 确认类别为SegmentKind.PARAGRAPH。


def test_horizontal_rule_separates_paragraphs():  # 验证水平分隔线断开段落。
    segments = _parse("above\n\n---\n\nbelow\n")  # 将结构片段列表设为解析响应的结果。

    texts = [s.text for s in segments if s.kind is SegmentKind.PARAGRAPH]  # 将待编码文本设为从结构片段列表中提取的各项。
    assert texts == ["above", "below"]  # 确认待编码文本等于列表。


def test_chinese_text_without_blank_lines_is_preserved():  # 中文段落常省略空行，解析器仍应连续收集正文。
    """中文段落常省略空行，解析器仍应连续收集正文。"""
    segments = _parse("## 标题\n这是第一句。\n这是第二句。\n")  # 将结构片段列表设为解析响应的结果。

    paragraphs = [s for s in segments if s.kind is SegmentKind.PARAGRAPH]  # 将段落列表设为从结构片段列表中提取的各项。
    assert len(paragraphs) == 1  # 确认段落列表的数量等于1。
    assert "第一句" in paragraphs[0].text and "第二句" in paragraphs[0].text  # 确认“第一句”包含在文本且“第二句”包含在文本。


def test_breadcrumb_property_joins_path():  # 验证位置提示按顺序连接各级标题。
    segments = _parse("# A\n\n## B\n\ntext\n")  # 将结构片段列表设为解析响应的结果。

    paragraph = [s for s in segments if s.kind is SegmentKind.PARAGRAPH][0]  # 将paragraph设为从结构片段列表中提取的各项中对应的项。
    assert paragraph.breadcrumb == "A > B"  # 确认标题路径等于“A > B”。


def test_parse_handles_bytes_and_reports_metadata():  # 验证字节输入可解析且生成元数据。
    document = parse_markdown("notes.md", "# T\n\nbody\n".encode())  # 将文档设为解析的结果。

    assert document.name == "notes.md"  # 确认名称等于“notes.md”。
    assert document.page_count == 1  # 确认总页数等于1。
    assert document.char_count > 0  # 确认字符数大于0。
    assert document.metadata["format"] == "markdown"  # 确认文档的元数据中的format等于“markdown”。
    assert not document.is_empty()  # 确认不满足“为空的结果”。


def test_empty_document_is_reported_empty():  # 验证空文档被识别为空。
    assert parse_markdown("empty.md", b"").is_empty()  # 确认为空的结果。