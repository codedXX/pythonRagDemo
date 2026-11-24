"""用 PyMuPDF 从 PDF 版面提取文本，以保留中文等多字节文字和页面位置信息。

按每页平均可提取字符数识别缺少可用文本层的扫描件。只有跨页重复且位于页边距内的文字才视为页眉或页脚，避免误删模板文档中重复的正文。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import re  # 引入正则表达式模块。
from collections import Counter  # 从 collections 导入Counter。
from dataclasses import dataclass  # 从 dataclasses 导入数据类装饰器。

import pymupdf  # 使用当前的模块导入名称。

from rag_demo.parsers.base import (  # 从 rag_demo.parsers.base 引入本模块使用的类型和工具。
    ParsedDocument,  # 导入解析文档。
    ScannedDocumentError,  # 导入缺少文本层异常。
    SegmentKind,  # 导入片段类型。
    StructuralSegment,  # 导入结构片段。
)  # 结束当前多行表达式。

# 每页平均可提取字符数达到此值时，才认为文本层可用。
_SENTENCE_END = re.compile(r"[。！？.!?;；:：]\s*$")  # 将_SENTENCE_END设为编译正则表达式的结果。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class _Line:  # 保存文本行及其在页面上的位置，用于识别页眉和页脚。
    """保存文本行及其在页面上的位置，用于识别页眉和页脚。"""

    text: str  # 声明文本字段。
    y_ratio: float  # 声明y_ratio字段。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class PdfParser:  # 提取 PDF 段落，并记录每个片段所在的页码。
    """提取 PDF 段落，并记录每个片段所在的页码。"""

    min_chars_per_page: int = 40  # 设置最低平均字符数字段的默认值。
    # 以出现该行的页数占比判断是否反复出现。
    furniture_page_fraction: float = 0.4  # 设置furniture_page_fraction字段的默认值。
    furniture_min_pages: int = 3  # 设置furniture_min_pages字段的默认值。
    # 只在页面顶部和底部的窄带内识别页眉页脚；边界过宽会吞掉正文首行。
    furniture_margin_ratio: float = 0.08  # 设置furniture_margin_ratio字段的默认值。

    def parse(self, name: str, content: bytes) -> ParsedDocument:  # 定义解析的处理逻辑。
        try:  # 在此捕获后续操作可能产生的异常。
            document = pymupdf.open(stream=content, filetype="pdf")  # 从上传字节直接打开 PDF 文档。
        except Exception as exc:  # 捕获通用异常。
            raise ValueError(f"cannot open PDF: {exc}") from exc  # 向上抛出值错误。

        try:  # 在此捕获后续操作可能产生的异常。
            page_count = document.page_count  # 取出当前使用的总页数。
            page_texts: list[str] = []  # 初始化各页正文为空列表。
            page_lines: list[list[_Line]] = []  # 初始化各页文本行为空列表。
            for index in range(page_count):  # 遍历指定的下标范围，逐项取得索引。
                page = document.load_page(index)  # 将页码设为load_page的结果。
                page_texts.append(page.get_text("text"))  # 保存页面纯文本，用于密度校验和段落提取。
                page_lines.append(self._lines_of(page))  # 另外提取文本行位置，用于识别页眉页脚。
        finally:  # 无论前面是否成功，都执行收尾操作。
            document.close()  # 无论提取是否成功都释放文档资源。

        total_chars = sum(len(t.strip()) for t in page_texts)  # 累计各页去掉首尾空白后的字符数。
        if page_count > 0:  # 当总页数大于0时进入此分支。
            density = total_chars / page_count  # 以全篇的平均值衡量文本层密度。
            if density < self.min_chars_per_page:  # 平均可提取字符过少时按扫描件处理。
                raise ScannedDocumentError(  # 向上抛出缺少文本层异常。
                    f"PDF appears to be image-based: {total_chars} extractable "  # 说明密度检测将文件识别为疑似扫描件。
                    f"characters across {page_count} page(s), which is "  # 给出页数及提取到的字符总数。
                    f"{density:.0f} per page (minimum {self.min_chars_per_page}). "  # 对照每页实测密度及配置阈值。
                    "Such a file would need OCR, which is not supported."  # 提示此类输入需要额外的文字识别步骤。
                )  # 结束当前多行表达式。

        furniture = self._detect_furniture(page_lines)  # 用页内位置和跨页频率同时识别页眉页脚。
        segments = self._segments_from(page_texts, furniture)  # 排除重复边缘文字后构造各页正文片段。

        # 如果过滤后没有正文，则保留原内容，避免误判导致整份文档为空。
        if not segments and furniture:  # 若过滤导致整份文档没有正文，则重新提取。
            segments = self._segments_from(page_texts, set())  # 使用空过滤集合，以保留可能被误判的内容。
            furniture = set()  # 清除已删除标记，使统计与实际输出一致。

        return ParsedDocument(  # 返回解析文档的结果。
            name=name,  # 传入名称参数。
            segments=segments,  # 传入结构片段列表参数。
            page_count=page_count,  # 传入总页数参数。
            char_count=total_chars,  # 传入字符数参数。
            metadata={"format": "pdf", "removed_furniture": len(furniture)},  # 传入元数据参数。
        )  # 结束当前多行表达式。

    # 页面解析。

    def _lines_of(self, page: "pymupdf.Page") -> list[_Line]:  # 提取各行文本和它们相对于页高的纵向位置。
        """提取各行文本和它们相对于页高的纵向位置。"""
        height = page.rect.height or 1.0  # 取得页高，异常零高度使用一作为除数。
        lines: list[_Line] = []  # 初始化文本行列表为空列表。
        try:  # 在此捕获后续操作可能产生的异常。
            payload = page.get_text("dict")  # 将载荷设为提取页面文本的结果。
        except Exception:  # 捕获通用异常。
            return lines  # 返回文本行列表。
        for block in payload.get("blocks", []):  # 遍历载荷中的块列表，逐项取得block。
            for line in block.get("lines", []):  # 遍历block中的文本行列表，逐项取得文本行。
                text = "".join(span.get("text", "") for span in line.get("spans", []))  # 把同一行中按字体样式分开的文本依次拼接。
                if not text.strip():  # 当不满足“清理首尾空白后的文本”时进入此分支。
                    continue  # 跳过本项，继续处理下一项。
                y0 = line.get("bbox", [0, 0, 0, 0])[1]  # 取包围矩形的顶部纵坐标。
                lines.append(_Line(text=text.strip(), y_ratio=y0 / height))  # 用行顶坐标除以页高，使不同纸张尺寸可以共用阈值。
        return lines  # 返回文本行列表。

    def _detect_furniture(self, page_lines: list[list[_Line]]) -> set[str]:  # 在页面边缘寻找跨页重复的文字。
        """在页面边缘寻找跨页重复的文字。"""
        if len(page_lines) < self.furniture_min_pages:  # 页数过少时无法可靠判断跨页重复。
            return set()  # 返回空集合。

        counts: Counter[str] = Counter()  # 将按状态计数设为Counter的结果。
        for lines in page_lines:  # 遍历各页文本行，逐项取得文本行列表。
            in_margin = {  # 用集合收集当前页位于边缘带中的文本。
                line.text  # 传入line.text。
                for line in lines  # 逐项处理推导式中的文本行。
                if line.y_ratio <= self.furniture_margin_ratio  # 选取页面顶部边缘带中的行。
                or line.y_ratio >= 1.0 - self.furniture_margin_ratio  # 同时选取页面底部边缘带中的行。
            }  # 结束本组字段。
            # 同一行在每页最多计一次，以免单页的重复文字被误判为页眉。
            counts.update(in_margin)  # 以页为单位累计每种边缘文本的出现次数。

        threshold = max(  # 重复页数需同时满足最低页数和页数比例。
            self.furniture_min_pages,  # 传入self.furniture_min_pages。
            int(len(page_lines) * self.furniture_page_fraction),  # 将文档页数乘以配置比例并取整数。
        )  # 结束当前多行表达式。
        return {text for text, count in counts.items() if count >= threshold}  # 只过滤跨页次数达到阈值的边缘文字。

    def _segments_from(  # 定义整理页面片段的处理逻辑。
        self, page_texts: list[str], furniture: set[str]  # 组合当前结果所需的各项数据。
    ) -> list[StructuralSegment]:  # 结束参数声明，进入函数或语句主体。
        segments: list[StructuralSegment] = []  # 初始化结构片段列表为空列表。
        for index, page_text in enumerate(page_texts, start=1):  # 从一开始赋予对外展示的页码。
            for paragraph in self._paragraphs(page_text, furniture):  # 遍历段落列表的结果，逐项取得paragraph。
                segments.append(  # 将结构片段的结果加入结构片段列表。
                    StructuralSegment(  # 为结构片段提供输入。
                        text=paragraph,  # 传入文本参数。
                        kind=SegmentKind.PARAGRAPH,  # 传入类别参数。
                        page=index,  # 将当前页码保存在每个段落上，以便引用。
                        heading_path=(),  # 传入标题路径参数。
                    )  # 结束当前多行表达式。
                )  # 结束当前多行表达式。
        return segments  # 返回结构片段列表。

    def _paragraphs(self, page_text: str, furniture: set[str]) -> list[str]:  # 移除页眉页脚和页码后，将页面文字行归并为段落。
        """移除页眉页脚和页码后，将页面文字行归并为段落。"""
        paragraphs: list[str] = []  # 初始化段落列表为空列表。
        buffer: list[str] = []  # 初始化缓冲区为空列表。

        def flush() -> None:  # 输出当前缓冲的内容，为下一段做准备。
            if buffer:  # 当缓冲区有效时进入此分支。
                joined = _join_lines(buffer)  # 将拼接结果设为_join_lines的结果。
                if joined.strip():  # 当清理首尾空白后的拼接结果时进入此分支。
                    paragraphs.append(joined.strip())  # 将清理首尾空白后的拼接结果加入段落列表。
                buffer.clear()  # 段落写入后清空当前缓冲。

        for raw_line in page_text.split("\n"):  # 遍历切分的结果，逐项取得当前原始行。
            line = raw_line.strip()  # 将文本行设为清理首尾空白后的当前原始行。
            if not line:  # 当文本行为空或未启用时进入此分支。
                flush()  # 输出当前缓冲片段。
                continue  # 跳过本项，继续处理下一项。
            if line in furniture:  # 当文本行包含在页眉页脚时进入此分支。
                continue  # 跳过本项，继续处理下一项。
            if _is_page_number(line):  # 当判断是否为页码的结果时进入此分支。
                continue  # 跳过本项，继续处理下一项。
            buffer.append(line)  # 将文本行加入缓冲区。
            # 遇到句末标点后又出现较长文本时，将其作为可能的段落边界。
            if _SENTENCE_END.search(line) and len(" ".join(buffer)) > 400:  # 正文累计超过四百字符且当前行像句末时切段。
                flush()  # 输出当前缓冲片段。

        flush()  # 输出当前缓冲片段。
        return paragraphs  # 返回段落列表。


def _is_page_number(line: str) -> bool:  # 识别单独成行的页码以及带装饰符的页码。
    """识别单独成行的页码以及带装饰符的页码。"""
    stripped = line.strip()  # 将去除首尾空白的文本设为清理首尾空白后的文本行。
    if not stripped or len(stripped) > 20:  # 当去除首尾空白的文本为假或去除首尾空白的文本的长度大于20时进入此分支。
        return False  # 返回假。
    return bool(re.fullmatch(r"[-–—\s]*(?:\d{1,4}|[ivxlcIVXLC]{1,6})[-–—\s]*", stripped))  # 整行须匹配阿拉伯或罗马页码，可带横线和空白。


def _join_lines(lines: list[str]) -> str:  # 拼接换行文本。
    """拼接换行文本；中文直接连接，拉丁文字之间按需补空格，避免单词粘连。"""
    if not lines:  # 当文本行列表为空或未启用时进入此分支。
        return ""  # 返回空字符串。
    result = lines[0]  # 将结果设为文本行列表中对应的项。
    for line in lines[1:]:  # 遍历文本行列表的指定区间，逐项取得文本行。
        if _needs_space(result, line):  # 根据相邻行的末字与首字决定是否补空格。
            result += " " + line  # 将“ ”加上文本行累加到结果。
        else:  # 处理其余情况。
            result += line  # 将文本行累加到结果。
    return result  # 返回结果。


def _needs_space(left: str, right: str) -> bool:  # 定义判断行间是否需要空格的处理逻辑。
    if not left or not right:  # 当左向量为假或右向量为假时进入此分支。
        return False  # 返回假。
    last = left[-1]  # 将末项设为左向量中对应的项。
    first = right[0]  # 将首项设为右向量中对应的项。
    # 行末连字符表示单词延续到下一行。
    if last == "-":  # 断行连字符已经连接单词，此处直接接上下行。
        return False  # 返回假。
    ascii_left = last.isascii() and (last.isalnum() or last in ",.;:!?)")  # 检测左侧是否为拉丁字母、数字或常见西文标点。
    ascii_right = first.isascii() and first.isalnum()  # 检测右侧是否以拉丁字母或数字开头。
    return ascii_left and ascii_right  # 同时符合两侧条件才在换行处插入空格。