"""按 Markdown 块结构解析文档。各片段继承所属标题路径，围栏代码整体保留。段落随块边界结束，因此标题和中文正文之间没有空行也能正确识别。"""

from __future__ import annotations  # 延后求值函数与字段的类型注解。

import re  # 使用正则表达式识别 Markdown 块边界。

from rag_demo.parsers.base import (  # 从共享解析模块导入文档结构类型。
    ParsedDocument,  # 返回统一的文档解析结果。
    SegmentKind,  # 用枚举区分标题、段落等块类型。
    StructuralSegment,  # 为每个块记录正文与标题路径。
)  # 完成解析类型导入。

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")  # 匹配一至六级标题并捕获标题文字。
_FENCE = re.compile(r"^\s*(```|~~~)")  # 识别反引号或波浪号组成的代码围栏。
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")  # 识别项目符号和数字序号开头的列表项。
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")  # 识别由竖线包围的表格行。
_QUOTE = re.compile(r"^\s*>\s?")  # 匹配行首引用符号及其后的可选空白。
_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$")  # 识别至少三个相同标记组成的水平分隔线。


def parse_markdown(name: str, content: bytes) -> ParsedDocument:  # 把 Markdown 文件解析成带标题路径的片段。
    text = content.decode("utf-8", errors="replace")  # 按 UTF-8 解码并以替代字符保留损坏位置。
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")  # 统一换行符后逐行处理。

    segments: list[StructuralSegment] = []  # 按原文顺序收集结构片段。
    # 每级标题占一个位置，下标零对应一级标题。
    heading_stack: list[str] = []  # 用栈保存当前各级标题。
    buffer: list[str] = []  # 暂存尚未形成片段的各行。
    buffer_kind = SegmentKind.PARAGRAPH  # 从普通段落的块类型开始。
    in_fence = False  # 初始位置在代码围栏之外。
    fence_marker = ""  # 记录打开代码块时使用的围栏样式。

    def flush() -> None:  # 将当前缓存提交为一个结构片段。
        nonlocal buffer, buffer_kind  # 允许在提交后重置外层缓存及其类型。
        body = "\n".join(buffer).strip("\n")  # 用换行符拼回正文并去掉首尾换行。
        if body.strip():  # 仅为有实际内容的缓存生成片段。
            segments.append(  # 把完成的块追加到解析结果。
                StructuralSegment(  # 构建携带结构信息的正文片段。
                    text=body,  # 保存块内拼接后的文本。
                    kind=buffer_kind,  # 沿用当前缓存对应的 Markdown 类型。
                    depth=0,  # 正文块的标题深度记为零。
                    heading_path=tuple(heading_stack),  # 保存当前标题路径的快照。
                )  # 完成正文片段构造。
            )  # 将片段加入结果列表。
        buffer = []  # 清空本次已经提交的行。
        buffer_kind = SegmentKind.PARAGRAPH  # 恢复默认段落类型，等待下一个块。

    for raw_line in lines:  # 依次识别每一行的块类型。
        # 逐行原样保留围栏内的内容，包括空行和形似标题的行。
        fence_match = _FENCE.match(raw_line)  # 检查本行是否以围栏标记开头。
        if in_fence:  # 已在围栏内时优先按代码处理。
            buffer.append(raw_line)  # 保留代码行原本的空白与标记。
            if fence_match and raw_line.strip().startswith(fence_marker):  # 遇到与起始样式相同的围栏时结束代码块。
                in_fence = False  # 退出围栏模式。
                flush()  # 提交包含结束围栏的完整代码块。
            continue  # 跳过普通 Markdown 块的识别。
        if fence_match:  # 围栏外遇到标记时开始新代码块。
            flush()  # 先提交围栏前的正文。
            in_fence = True  # 进入逐行保留代码的模式。
            fence_marker = fence_match.group(1)  # 记住用于识别结束行的标记。
            buffer_kind = SegmentKind.CODE  # 将接下来缓存的片段标为代码。
            buffer = [raw_line]  # 将起始围栏作为代码块首行。
            continue  # 从下一行开始收集代码。

        heading = _HEADING.match(raw_line)  # 提取可能的标题标记与文字。
        if heading:  # 单独处理标题行。
            flush()  # 在切换标题路径前提交旧块。
            level = len(heading.group(1))  # 用井号数量确定标题级别。
            title = heading.group(2).strip()  # 提取去掉首尾空白的标题文本。
            # 保留上级标题，再将当前标题加入路径。
            heading_stack = heading_stack[: level - 1]  # 截去旧的同级标题及其子标题。
            while len(heading_stack) < level - 1:  # 处理从浅层直接跳到深层的标题。
                heading_stack.append("")  # 为空缺的层级保留位置。
            heading_stack.append(title)  # 将当前标题放入对应深度。
            segments.append(  # 把标题自身也保存为结构片段。
                StructuralSegment(  # 构建标题类型的片段。
                    text=raw_line.strip(),  # 保留标题的原始 Markdown 标记。
                    kind=SegmentKind.HEADING,  # 标记为标题以便分块器识别。
                    depth=level,  # 记录由井号数确定的标题级别。
                    heading_path=tuple(h for h in heading_stack if h),  # 输出标题路径时过滤跳级留下的空位。
                )  # 完成标题片段构造。
            )  # 将标题片段加入解析结果。
            continue  # 标题已独立处理，开始下一行。

        if _HR.match(raw_line):  # 水平线作为块之间的边界。
            flush()  # 提交分隔线之前的内容。
            continue  # 跳过分隔线本身。

        if not raw_line.strip():  # 把空白行识别为块边界。
            flush()  # 提交空白行之前积累的块。
            continue  # 跳过空白行本身。

        if _QUOTE.match(raw_line):  # 识别以引用符开头的内容。
            if buffer and buffer_kind is not SegmentKind.QUOTE:  # 进入引用前检查是否有不同类型的块待提交。
                flush()  # 先提交前一个非引用块。
            buffer_kind = SegmentKind.QUOTE  # 将后续缓存标记为引用。
            buffer.append(_QUOTE.sub("", raw_line))  # 去掉行首引用符后保留正文。
            continue  # 本行引用处理完毕。

        if _TABLE_ROW.match(raw_line):  # 识别用竖线包围的表格行。
            if buffer and buffer_kind is not SegmentKind.TABLE:  # 进入表格前检查是否要结束前一块。
                flush()  # 提交前一个非表格块。
            buffer_kind = SegmentKind.TABLE  # 将连续表格行归为表格片段。
            buffer.append(raw_line)  # 保留表格行及其单元格分隔符。
            continue  # 本行表格处理完毕。

        if _LIST_ITEM.match(raw_line):  # 识别带项目符或序号的列表行。
            if buffer and buffer_kind is not SegmentKind.LIST:  # 进入列表前检查是否要结束前一块。
                flush()  # 提交前一个非列表块。
            buffer_kind = SegmentKind.LIST  # 将连续列表内容标为列表片段。
            buffer.append(raw_line)  # 保留列表项的标记及正文。
            continue  # 本行列表处理完毕。

        # 根据当前打开的块，将普通文本并入列表或段落。
        if buffer_kind is SegmentKind.CODE:  # 检查普通文本是否仍带有代码块类型。
            buffer_kind = SegmentKind.PARAGRAPH  # 让后续普通文本恢复段落类型。
        buffer.append(raw_line)  # 将未被前面规则消费的文本并入当前块。

    flush()  # 在文件末尾提交最后一块。
    return ParsedDocument(  # 将所有块组装为统一的文档结果。
        name=name,  # 保留文档的来源文件名。
        segments=segments,  # 交付按原文顺序收集的片段。
        page_count=1,  # Markdown 按单页逻辑文档计数。
        char_count=sum(len(s.text) for s in segments),  # 累加各结构片段实际保留的字符数。
        metadata={"format": "markdown"},  # 注明解析结果来自 Markdown。
    )  # 完成文档结果构造。
