"""用 markdown-it-py 识别顶层块，保留原文及其所属章节。"""

from markdown_it import MarkdownIt  # 引入按 CommonMark 规则识别 Markdown 结构的解析器。

from rag_demo.parsers.base import ParsedDocument, SegmentKind, StructuralSegment  # 使用入库流程共用的文档、片段类别和片段类型。


# 让链接定义也带上源行范围，保留文档中可用于检索的链接目标。
_PARSER = MarkdownIt("commonmark", {"inline_definitions": True}).enable("table")  # 以 CommonMark 为基础启用表格规则，并让链接定义生成独立节点。
_BLOCK_KINDS = {  # 将解析器的块节点名称映射到分块器使用的片段类别。
    "heading_open": SegmentKind.HEADING,  # 井号标题和下划线式标题都由标题起始节点标识。
    "paragraph_open": SegmentKind.PARAGRAPH,  # 从段落起始节点定位整段正文。
    "html_block": SegmentKind.PARAGRAPH,  # 块级 HTML 的原文也作为可检索的正文。
    "definition": SegmentKind.PARAGRAPH,  # 把链接定义保留为正文，让引用目标地址也能被检索。
    "bullet_list_open": SegmentKind.LIST,  # 从无序列表的起始节点截取整份列表。
    "ordered_list_open": SegmentKind.LIST,  # 有序列表和无序列表共用列表类别。
    "blockquote_open": SegmentKind.QUOTE,  # 从引用的起始节点截取引文及其嵌套内容。
    "table_open": SegmentKind.TABLE,  # 从表格起始节点截取表头、分隔行与数据行。
    "fence": SegmentKind.CODE,  # 将围栏及其内部内容作为一个代码片段。
    "code_block": SegmentKind.CODE,  # 缩进形成的代码块也归入代码类别。
}  # 完成会被保留的块类型映射。


def parse_markdown(name: str, content: bytes) -> ParsedDocument:  # 将上传文件的字节内容解析成带章节路径的文档片段。
    text = content.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")  # 用 UTF-8 解码，替换无效字节，并统一三种常见换行符。
    lines = text.split("\n")  # 按统一后的换行符拆分原文，供后续按行号切片。
    tokens = _PARSER.parse(text)  # 解析整篇文档，取得带嵌套层级和源行范围的节点序列。
    headings: list[tuple[int, str]] = []  # 按从浅到深的顺序保存当前章节各级标题的深度和文字。
    segments: list[StructuralSegment] = []  # 按源文档顺序收集将交给分块器的片段。

    for index, token in enumerate(tokens):  # 遍历节点时保留序号，以便在标题处读取紧随其后的文字。
        kind = _BLOCK_KINDS.get(token.type)  # 查出当前节点对应的片段类别，未列入映射的类型得到空值。
        if token.level != 0 or kind is None:  # 仅选择受支持的顶层块，嵌套内容已包含在外层块的原文中。
            continue  # 跳过嵌套节点及没有片段映射的节点。

        depth = 0  # 普通块深度记为零，章节归属由标题路径单独记录。
        if kind is SegmentKind.HEADING:  # 遇到顶层标题时先切换章节，再生成该标题的片段。
            depth = int(token.tag[1:])  # 从 h1 至 h6 标签的数字部分读出标题深度。
            while headings and headings[-1][0] >= depth:  # 同级新标题会替换旧标题，更浅的新标题还会结束原有子节。
                headings.pop()  # 从路径末端移除已经结束的章节。
            # heading_open 后的 inline 节点保存去除标题标记的文字。
            title = tokens[index + 1].content.strip()  # 读取标题正文并去掉首尾空白，作为路径中的章节名称。
            if title:  # 只将有文字的标题加入章节路径。
                headings.append((depth, title))  # 将本节的深度和名称接在仍然有效的父级路径之后。

        start, end = token.map  # 取得块的起始行和结束边界，结束边界指向块后的第一行。
        segments.append(  # 按节点出现顺序将当前块加入结果。
            StructuralSegment(  # 用原文、类别和来源位置构造一个结构片段。
                text="\n".join(lines[start:end]),  # 按左闭右开的行范围拼回原文，保留围栏和列表缩进等标记。
                kind=kind,  # 保存由节点类型映射出的片段类别。
                depth=depth,  # 记录标题级别，非标题块沿用零值。
                heading_path=tuple(title for _, title in headings),  # 复制当前标题名称序列，让后续章节切换不影响已有片段。
            )  # 完成当前结构片段的构造。
        )  # 将该片段追加到文档的有序片段列表。

    return ParsedDocument(  # 将解析结果组装为入库流程使用的文档对象。
        name=name,  # 保留上传文件名供引用来源使用。
        segments=segments,  # 传递按原文顺序生成的结构片段。
        page_count=1,  # 按项目约定将 Markdown 文档计为一页。
        char_count=sum(len(segment.text) for segment in segments),  # 累加实际保留的片段文本长度作为文档字符数。
        metadata={"format": "markdown"},  # 标记源格式，便于下游识别 Markdown 文档。
    )  # 返回包含片段及统计信息的文档。
