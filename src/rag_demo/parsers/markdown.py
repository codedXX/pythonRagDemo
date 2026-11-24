"""按 Markdown 块结构解析文档。各片段继承所属标题路径，围栏代码整体保留。段落随块边界结束，因此标题和中文正文之间没有空行也能正确识别。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import re  # 引入正则表达式模块。
from dataclasses import dataclass  # 从 dataclasses 导入数据类装饰器。

from rag_demo.parsers.base import (  # 从 rag_demo.parsers.base 引入本模块使用的类型和工具。
    ParsedDocument,  # 导入解析文档。
    SegmentKind,  # 导入片段类型。
    StructuralSegment,  # 导入结构片段。
)  # 结束当前多行表达式。

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")  # 分别捕获标题标记与标题正文，允许末尾带井号。
_FENCE = re.compile(r"^\s*(```|~~~)")  # 识别反引号或波浪线组成的代码围栏。
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")  # 识别无序符号或序号开头的列表项。
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")  # 识别以竖线划分单元格的表格行。
_QUOTE = re.compile(r"^\s*>\s?")  # 匹配引文前缀，供后续从正文中移除。
_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$")  # 识别由至少三个同类标记组成的水平分隔线。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class MarkdownParser:  # 将上传的 Markdown 内容转换为结构化片段。
    """将上传的 Markdown 内容转换为结构化片段。"""

    def parse(self, name: str, content: bytes) -> ParsedDocument:  # 定义解析的处理逻辑。
        text = content.decode("utf-8", errors="replace")  # 按统一字符编码解码上传内容，用替代字符保留可读部分。
        segments = self.parse_text(text)  # 将结构片段列表设为parse_text的结果。
        return ParsedDocument(  # 返回解析文档的结果。
            name=name,  # 传入名称参数。
            segments=segments,  # 传入结构片段列表参数。
            page_count=1,  # 标记文档没有固定分页，此处按一页登记。
            char_count=sum(len(s.text) for s in segments),  # 传入字符数参数。
            metadata={"format": "markdown"},  # 传入元数据参数。
        )  # 结束当前多行表达式。

    def parse_text(self, text: str) -> list[StructuralSegment]:  # 解析文本块，并跟踪各级标题组成的路径。
        """解析文本块，并跟踪各级标题组成的路径。"""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")  # 将不同系统的换行符统一后按行解析。

        segments: list[StructuralSegment] = []  # 初始化结构片段列表为空列表。
        # 每级标题占一个位置，下标零对应一级标题。
        heading_stack: list[str] = []  # 初始化各级标题栈为空列表。
        buffer: list[str] = []  # 初始化缓冲区为空列表。
        buffer_kind = SegmentKind.PARAGRAPH  # 将缓冲片段类型设为SegmentKind.PARAGRAPH。
        in_fence = False  # 将围栏内标志设为假。
        fence_marker = ""  # 将围栏标记设为空字符串。

        def flush() -> None:  # 输出当前缓冲的内容，为下一段做准备。
            nonlocal buffer, buffer_kind  # 在嵌套函数中更新外层的缓冲区、缓冲片段类型。
            body = "\n".join(buffer).strip("\n")  # 按行恢复片段正文，保留内部空行。
            if body.strip():  # 当清理首尾空白后的正文时进入此分支。
                segments.append(  # 将结构片段的结果加入结构片段列表。
                    StructuralSegment(  # 为结构片段提供输入。
                        text=body,  # 传入文本参数。
                        kind=buffer_kind,  # 传入类别参数。
                        depth=0,  # 传入标题层级参数。
                        heading_path=tuple(heading_stack),  # 保存当前标题路径的快照，避免后续标题变更影响已建片段。
                    )  # 结束当前多行表达式。
                )  # 结束当前多行表达式。
            buffer = []  # 初始化缓冲区为空列表。
            buffer_kind = SegmentKind.PARAGRAPH  # 将缓冲片段类型设为SegmentKind.PARAGRAPH。

        for raw_line in lines:  # 遍历文本行列表，逐项取得当前原始行。
            # 逐行原样保留围栏内的内容，包括空行和形似标题的行。
            fence_match = _FENCE.match(raw_line)  # 检查当前行是否出现代码围栏。
            if in_fence:  # 当围栏内标志有效时进入此分支。
                buffer.append(raw_line)  # 将当前原始行加入缓冲区。
                if fence_match and raw_line.strip().startswith(fence_marker):  # 只有与开头标记相符的围栏才结束代码块。
                    in_fence = False  # 将围栏内标志设为假。
                    flush()  # 输出当前缓冲片段。
                continue  # 跳过本项，继续处理下一项。
            if fence_match:  # 当围栏匹配结果有效时进入此分支。
                flush()  # 输出当前缓冲片段。
                in_fence = True  # 将围栏内标志设为真。
                fence_marker = fence_match.group(1)  # 记下是反引号还是波浪线，供匹配结尾使用。
                buffer_kind = SegmentKind.CODE  # 将缓冲片段类型设为SegmentKind.CODE。
                buffer = [raw_line]  # 把起始围栏本身也保留在代码片段中。
                continue  # 跳过本项，继续处理下一项。

            heading = _HEADING.match(raw_line)  # 将标题设为模式匹配的结果。
            if heading:  # 当标题有效时进入此分支。
                flush()  # 输出当前缓冲片段。
                level = len(heading.group(1))  # 以井号数量判断标题层级。
                title = heading.group(2).strip()  # 取得标题正文并清除首尾空白。
                # 保留上级标题，再将当前标题加入路径。
                heading_stack = heading_stack[: level - 1]  # 保留上级路径，移除旧的同级和下级标题。
                while len(heading_stack) < level - 1:  # 遇到跳级标题时补齐缺失的层级。
                    heading_stack.append("")  # 先用空字符串占位，输出路径时再过滤。
                heading_stack.append(title)  # 把当前标题接到上级路径之后。
                segments.append(  # 将结构片段的结果加入结构片段列表。
                    StructuralSegment(  # 为结构片段提供输入。
                        text=raw_line.strip(),  # 传入文本参数。
                        kind=SegmentKind.HEADING,  # 传入类别参数。
                        depth=level,  # 传入标题层级参数。
                        heading_path=tuple(h for h in heading_stack if h),  # 只保留真实存在的标题，滤除跳级占位。
                    )  # 结束当前多行表达式。
                )  # 结束当前多行表达式。
                continue  # 跳过本项，继续处理下一项。

            if _HR.match(raw_line):  # 水平线作为分段信号，不写入段落正文。
                flush()  # 输出当前缓冲片段。
                continue  # 跳过本项，继续处理下一项。

            if not raw_line.strip():  # 当不满足“清理首尾空白后的当前原始行”时进入此分支。
                flush()  # 输出当前缓冲片段。
                continue  # 跳过本项，继续处理下一项。

            if _QUOTE.match(raw_line):  # 当模式匹配的结果时进入此分支。
                if buffer and buffer_kind is not SegmentKind.QUOTE:  # 当缓冲区且缓冲片段类型不为SegmentKind.QUOTE时进入此分支。
                    flush()  # 输出当前缓冲片段。
                buffer_kind = SegmentKind.QUOTE  # 将缓冲片段类型设为SegmentKind.QUOTE。
                buffer.append(_QUOTE.sub("", raw_line))  # 去掉引用标记并保留引文本身。
                continue  # 跳过本项，继续处理下一项。

            if _TABLE_ROW.match(raw_line):  # 当模式匹配的结果时进入此分支。
                if buffer and buffer_kind is not SegmentKind.TABLE:  # 当缓冲区且缓冲片段类型不为SegmentKind.TABLE时进入此分支。
                    flush()  # 输出当前缓冲片段。
                buffer_kind = SegmentKind.TABLE  # 将缓冲片段类型设为SegmentKind.TABLE。
                buffer.append(raw_line)  # 将当前原始行加入缓冲区。
                continue  # 跳过本项，继续处理下一项。

            if _LIST_ITEM.match(raw_line):  # 当模式匹配的结果时进入此分支。
                if buffer and buffer_kind is not SegmentKind.LIST:  # 当缓冲区且缓冲片段类型不为SegmentKind.LIST时进入此分支。
                    flush()  # 输出当前缓冲片段。
                buffer_kind = SegmentKind.LIST  # 将缓冲片段类型设为SegmentKind.LIST。
                buffer.append(raw_line)  # 将当前原始行加入缓冲区。
                continue  # 跳过本项，继续处理下一项。

            # 根据当前打开的块，将普通文本并入列表或段落。
            if buffer_kind is SegmentKind.CODE:  # 当缓冲片段类型为SegmentKind.CODE时进入此分支。
                buffer_kind = SegmentKind.PARAGRAPH  # 将缓冲片段类型设为SegmentKind.PARAGRAPH。
            buffer.append(raw_line)  # 将当前原始行加入缓冲区。

        flush()  # 输出当前缓冲片段。
        return segments  # 返回结构片段列表。