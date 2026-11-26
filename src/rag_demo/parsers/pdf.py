"""用 PyMuPDF 从 PDF 版面提取文本，以保留中文等多字节文字和页面位置信息。

按每页平均可提取字符数识别缺少可用文本层的扫描件。只有跨页重复且位于页边距内的文字才视为页眉或页脚，避免误删模板文档中重复的正文。"""

from __future__ import annotations  # 延后求值解析器使用的类型注解。

import re  # 用正则识别句末标点与独立页码。
from collections import Counter  # 按页统计边缘文本出现的次数。

import pymupdf  # 使用 PyMuPDF 读取页面文字及坐标。

from rag_demo.parsers.base import (  # 引入 PDF 与其他解析器共用的结果类型。
    ParsedDocument,  # 用统一文档类型返回解析结果。
    ScannedDocumentError,  # 在可提取文本过少时报告扫描件错误。
    SegmentKind,  # 标注提取片段的结构类型。
    StructuralSegment,  # 携带每段的正文与来源页码。
)  # 完成共享类型导入。

MARGIN_RATIO = 0.08  # 取页面顶部和底部各 8% 作为边缘区域。
_SENTENCE_END = re.compile(r"[。！？.!?;；:：]\s*$")  # 识别可作为较长段落终点的行末标点。


def parse_pdf(name: str, content: bytes, min_chars_per_page: int = 40) -> ParsedDocument:  # 按页提取 PDF 文字并保留页码。
    try:  # 单独处理打开 PDF 时的失败。
        document = pymupdf.open(stream=content, filetype="pdf")  # 直接从上传的字节流打开 PDF。
    except Exception as exc:  # 捕获解析库在打开文件时抛出的异常。
        raise ValueError(f"cannot open PDF: {exc}") from exc  # 附上原始原因以便定位无效 PDF。

    with document:  # 在读取结束后释放 PDF 文档资源。
        page_count = document.page_count  # 记录总页数以计算平均文本密度。
        page_texts: list[str] = []  # 按页保存纯文本供后续分段。
        margin_counts: Counter[str] = Counter()  # 按跨页出现次数统计候选页眉和页脚。
        for page in document:  # 依次读取每一页。
            page_texts.append(page.get_text("text"))  # 保留本页的文本层内容。
            if page_count >= 3:  # 至少三页才可能达到重复文本的过滤门槛。
                margin_counts.update(_margin_texts(page))  # 对本页去重后的边缘文本各计一次。

    total_chars = sum(len(t.strip()) for t in page_texts)  # 累计各页去掉首尾空白后的字符数。
    if page_count > 0:  # 有页面时才计算每页平均字符数。
        density = total_chars / page_count  # 将全部可提取字符数平摊到每页。
        if density < min_chars_per_page:  # 比较实测文本密度与配置门槛。
            raise ScannedDocumentError(  # 向上游报告疑似图像型 PDF。
                f"PDF appears to be image-based: {total_chars} extractable "  # 在诊断中列出提取到的总字符数。
                f"characters across {page_count} page(s), which is "  # 附上用于计算密度的页数。
                f"{density:.0f} per page (minimum {min_chars_per_page}). "  # 显示实测平均值和最低要求。
                "Such a file would need OCR, which is not supported."  # 说明此类文件需要 OCR 才能提取文字。
            )  # 完成扫描件错误说明。

    # 一段边缘文字至少出现在三页，才可能是重复页眉或页脚。
    threshold = max(3, int(page_count * 0.4))  # 重复门槛取三页和总页数四成的整数部分中的较大值。
    furniture = {text for text, count in margin_counts.items() if count >= threshold}  # 选出达到门槛的边缘文本。
    segments = _segments_from(page_texts, furniture)  # 在分段时过滤这些重复文本。

    # 如果过滤后没有正文，则保留原内容，避免误判导致整份文档为空。
    if not segments and furniture:  # 检查过滤是否导致所有正文都消失。
        segments = _segments_from(page_texts, set())  # 取消边缘过滤重新提取片段。
        furniture = set()  # 将实际过滤的边缘文本数重置为零。

    return ParsedDocument(  # 返回包含来源与统计的解析结果。
        name=name,  # 保留原始文件名。
        segments=segments,  # 交付带页码的正文片段。
        page_count=page_count,  # 保留 PDF 的实际页数。
        char_count=total_chars,  # 记录边缘过滤前提取的字符数。
        metadata={"format": "pdf", "removed_furniture": len(furniture)},  # 注明文件格式及被过滤的不同边缘文本数。
    )  # 完成 PDF 文档结果组装。


def _margin_texts(page: pymupdf.Page) -> set[str]:  # 收集单页顶部和底部的候选文本。
    """每页只统计一次重复文字；缺少版面信息时仍可以使用纯文本。"""
    try:  # 尝试读取含坐标的页面结构。
        layout = page.get_text("dict")  # 用字典形式获取文字块、行和文本片段。
    except Exception:  # 页面无法提供布局时跳过边缘识别。
        return set()  # 将这一页视为没有候选边缘文本。

    texts = set()  # 用集合保证同一页内的同一文本只计一次。
    height = page.rect.height or 1.0  # 取页高作为坐标归一化的分母。
    for block in layout["blocks"]:  # 遍历本页的版面块。
        for line in block.get("lines", []):  # 遍历块内的文本行，跳过无文本行的块。
            y = line["bbox"][1] / height  # 把行上边缘纵坐标换算为页高比例。
            if y <= MARGIN_RATIO or y >= 1.0 - MARGIN_RATIO:  # 选取顶部或底部边缘范围内的行。
                text = "".join(span["text"] for span in line["spans"]).strip()  # 按顺序拼合行内各文本片段并清理空白。
                if text:  # 忽略只有空白的候选行。
                    texts.add(text)  # 保存该页出现过的边缘文本。
    return texts  # 返回当前页的去重结果。


def _segments_from(page_texts: list[str], furniture: set[str]) -> list[StructuralSegment]:  # 将各页文本组装为段落并保留出处。
    """在空行、较长句末或页面末尾结段，保留每段的页码。"""
    segments = []  # 收集所有页面的段落。
    for page_number, text in enumerate(page_texts, start=1):  # 从一开始编号以对应读者看到的页码。
        lines = []  # 单独缓存本页当前段落的各行。
        for raw in (text + "\n").split("\n"):  # 在页尾追加换行，确保最后一段也得到提交。
            line = raw.strip()  # 去掉当前行两端空白。
            if line:  # 非空行先通过页码和重复文本过滤。
                if line in furniture or _is_page_number(line):  # 识别重复边缘文本或独立页码。
                    continue  # 跳过这些行以免进入正文。
                lines.append(line)  # 将正文行并入当前段落。
                if not (_SENTENCE_END.search(line) and len(" ".join(lines)) > 400):  # 在句末且段长超过四百字符时才提前切段。
                    continue  # 继续累积尚未达到切段条件的正文。
            if lines:  # 在边界处提交已有正文。
                segments.append(StructuralSegment(  # 构建并保存属于当前页的段落。
                    text=_join_lines(lines), kind=SegmentKind.PARAGRAPH, page=page_number  # 按字符边界拼合正文并记录页码。
                ))  # 完成段落构造与追加。
                lines.clear()  # 清空缓存供下一段使用。
    return segments  # 返回按页序生成的段落。


def _is_page_number(line: str) -> bool:  # 判断整行是否仅由页码及其装饰组成。
    """识别单独成行的页码以及带装饰符的页码。"""
    stripped = line.strip()  # 去掉页码两侧的空白。
    if not stripped or len(stripped) > 20:  # 排除空文本和过长的正文行。
        return False  # 这些行不作为独立页码处理。
    return bool(re.fullmatch(r"[-–—\s]*(?:\d{1,4}|[ivxlcIVXLC]{1,6})[-–—\s]*", stripped))  # 完整匹配阿拉伯数字或限定长度的罗马数字及两侧装饰。


def _join_lines(lines: list[str]) -> str:  # 将同一段落的提取行拼回连续文本。
    """拼接换行文本；中文直接连接，拉丁文字之间按需补空格，避免单词粘连。"""
    if not lines:  # 处理没有正文行的段落。
        return ""  # 空段落返回空字符串。
    result = lines[0]  # 用首行初始化段落正文。
    for line in lines[1:]:  # 依次连接后续各行。
        if _needs_space(result, line):  # 检查相邻字符之间是否需要分词空格。
            result += " " + line  # 加一个空格以分开两行边界处的单词。
        else:  # 其余边界直接相连。
            result += line  # 保留中文或连字符处的连续性。
    return result  # 返回拼接后的整段文字。


def _needs_space(left: str, right: str) -> bool:  # 根据前一行末尾与下一行开头判定是否补空格。
    if not left or not right:  # 先排除缺少任一侧字符的情况。
        return False  # 缺少边界字符时无需补空格。
    last = left[-1]  # 取已拼正文的末字符。
    first = right[0]  # 取下一行的首字符。
    # 行末连字符表示单词延续到下一行。
    if last == "-":  # 识别行末用于连接单词的连字符。
        return False  # 保留连字符与下一行之间的紧邻关系。
    ascii_left = last.isascii() and (last.isalnum() or last in ",.;:!?)")  # 判断左侧是否为 ASCII 字母、数字或列出的标点。
    ascii_right = first.isascii() and first.isalnum()  # 判断右侧是否为 ASCII 字母或数字。
    return ascii_left and ascii_right  # 仅在两侧都符合条件时插入空格。
