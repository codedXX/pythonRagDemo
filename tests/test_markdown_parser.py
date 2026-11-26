"""验证 Markdown 的结构解析和标题路径。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

from rag_demo.parsers.base import SegmentKind  # 从 rag_demo.parsers.base 导入片段类型。
from rag_demo.parsers.markdown import parse_markdown  # 从 rag_demo.parsers.markdown 导入标记文本解析器。


def _parse(text: str):  # 让测试可以直接用 Markdown 文本构造解析样本。
    return parse_markdown("test.md", text.encode()).segments  # 模拟上传的文件名和字节内容，取出待断言的结构片段。


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

def test_setext_heading_and_sibling_paths():  # 验证下划线式一级标题和同级二级标题生成正确的章节路径。
    segments = _parse("安装指南\n===\n\n## 环境要求\n正文甲\n\n## 配置\n正文乙")  # 构造含一个一级标题、两个二级标题和各节独立正文的样本。
    assert [(s.kind, s.text, s.depth, s.heading_path) for s in segments] == [  # 按原文顺序逐段核对类别、文本、标题级别和完整路径。
        (SegmentKind.HEADING, "安装指南\n===", 1, ("安装指南",)),  # 一级标题原文保留等号下划线，路径只采用标题名称。
        (SegmentKind.HEADING, "## 环境要求", 2, ("安装指南", "环境要求")),  # 首个二级标题应接在安装指南之下。
        (SegmentKind.PARAGRAPH, "正文甲", 0, ("安装指南", "环境要求")),  # 第一段正文应继承环境要求的路径，自身标题深度为零。
        (SegmentKind.HEADING, "## 配置", 2, ("安装指南", "配置")),  # 第二个二级标题应在同一父级下替换环境要求。
        (SegmentKind.PARAGRAPH, "正文乙", 0, ("安装指南", "配置")),  # 第二段正文应归入配置章节。
    ]  # 上述五项也是期望的完整片段序列。


def test_short_inner_fence_keeps_example_in_its_section():  # 验证代码块内较短的围栏不会提前结束代码或改变所属章节。
    code = "````md\n```\n# 假标题\n\n````"  # 用四个反引号包住三个反引号和一行形似标题的示例代码。
    heading, example, after = _parse(f"# 示例\n\n{code}\n\n继续")  # 解析后应恰有标题、代码示例及后续正文三个片段。
    assert (example.kind, example.text) == (SegmentKind.CODE, code)  # 核对代码片段原样保留围栏、空行和内部井号。
    assert (after.kind, after.text) == (SegmentKind.PARAGRAPH, "继续")  # 核对外层围栏结束后的文字单独成为普通段落。
    assert heading.heading_path == example.heading_path == after.heading_path == ("示例",)  # 核对三个片段都属于示例章节，代码里的井号不会创建新路径。


def test_nested_quote_has_one_copy_of_its_source():  # 验证嵌套引用的原文只生成一份顶层片段。
    quote = "> # 引用内标题\n>\n> - 父项\n>   - 子项"  # 在引用内放置标题和嵌套列表以检验块边界。
    source = f"# 操作说明\n\n{quote}\n\n- 独立列表"  # 在引用外添加真实章节标题和独立列表。
    heading, quoted, listed = _parse(source)  # 按预期的三个顶层块解包，多生成嵌套片段也会使测试失败。
    assert (quoted.kind, quoted.text) == (SegmentKind.QUOTE, quote)  # 核对完整引文与所有嵌套标记保存在同一个引用片段中。
    assert (listed.kind, listed.text) == (SegmentKind.LIST, "- 独立列表")  # 核对引用结束后的列表独立成块。
    assert heading.heading_path == quoted.heading_path == listed.heading_path == ("操作说明",)  # 核对引用内的标题未改变引用本身及后续列表的章节路径。


def test_table_without_outer_pipes_and_standalone_pipe_text():  # 验证无外侧竖线的表格和仅有竖线的普通段落能被区分。
    table = "名称 | 数量\n--- | ---\n苹果 | 2"  # 构造仅以内部竖线分列、含表头分隔行的有效表格。
    parsed_table, paragraph = _parse(f"{table}\n\n| 普通文字 |")  # 在表格后另起一段加入带外侧竖线的文字。
    assert (parsed_table.kind, parsed_table.text) == (SegmentKind.TABLE, table)  # 核对有效表格整体被识别并保留原文。
    assert (paragraph.kind, paragraph.text) == (SegmentKind.PARAGRAPH, "| 普通文字 |")  # 核对缺少表头分隔行的文字保留为普通段落。
