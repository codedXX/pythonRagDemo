"""在测试中生成真实 PDF，覆盖生产解析路径，并用缺少文本层的页面验证扫描件处理。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pymupdf  # 引入版面解析库。
import pytest  # 引入测试框架。

from rag_demo.parsers.base import ScannedDocumentError  # 从 rag_demo.parsers.base 导入缺少文本层异常。
from rag_demo.parsers.pdf import parse_pdf, _is_page_number, _needs_space  # 从 rag_demo.parsers.pdf 引入本模块使用的类型和工具。


def _build_pdf(pages: list[str], header: str | None = None) -> bytes:  # 按输入的页面内容各创建一个文本框，可在页边距内绘制重复页眉。
    """按输入的页面内容各创建一个文本框，可在页边距内绘制重复页眉。"""
    document = pymupdf.open()  # 将文档设为打开文档的结果。
    for body in pages:  # 遍历页面列表，逐项取得正文。
        page = document.new_page()  # 将页码设为创建页面的结果。
        # 用可自动换行的文本框构造测试样本，覆盖解析器重新拼接段落的逻辑。
        page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), body, fontsize=11)  # 写入文本框。
        if header:  # 当页眉有效时进入此分支。
            page.insert_text((72, 30), header, fontsize=9)  # 调用insert_text。
    data = document.tobytes()  # 将数据设为导出文档字节串的结果。
    document.close()  # 关闭文档并释放资源。
    return data  # 返回数据。


def _blank_pdf(pages: int = 2) -> bytes:  # 定义_blank_pdf的处理逻辑。
    document = pymupdf.open()  # 将文档设为打开文档的结果。
    for _ in range(pages):  # 遍历指定的下标范围，逐项取得_。
        document.new_page()  # 空白页面用于模拟完全缺少文本层的扫描页。
    data = document.tobytes()  # 将数据设为导出文档字节串的结果。
    document.close()  # 关闭文档并释放资源。
    return data  # 返回数据。


def test_parses_pages_and_records_page_numbers():  # 验证解析多页文档并记录来源页码。
    data = _build_pdf(  # 将数据设为构造测试文档的结果。
        [  # 开始列出各项。
            "First page sentence number one, written long enough to be a real page of text.",  # 为第一页提供足以通过密度检查的正文。
            "Second page content, also long enough that the file is not mistaken for a scan.",  # 用不同的第二页正文验证来源页码。
        ]  # 结束列表。
    )  # 结束当前多行表达式。

    document = parse_pdf("sample.pdf", data)  # 将文档设为解析的结果。

    assert document.page_count == 2  # 确认总页数等于2。
    assert document.metadata["format"] == "pdf"  # 确认文档的元数据中的format等于“pdf”。
    pages_seen = {segment.page for segment in document.segments}  # 将pages_seen设为从结构片段列表中提取的各项。
    assert pages_seen == {1, 2}  # 确认pages_seen等于集合。


def test_text_is_extracted_in_readable_form():  # 验证提取的文本可读。
    data = _build_pdf(["The quick brown fox jumps over the lazy dog. " * 3])  # 将数据设为构造测试文档的结果。

    document = parse_pdf("sample.pdf", data)  # 将文档设为解析的结果。
    joined = " ".join(s.text for s in document.segments)  # 将拼接结果设为拼接后的从结构片段列表中提取的各项。

    assert "quick brown fox" in joined  # 确认“quick brown fox”包含在拼接结果。


def test_scanned_pdf_is_rejected():  # 缺少文本层的文件应被明确拒绝。
    """缺少文本层的文件应被明确拒绝。"""
    with pytest.raises(ScannedDocumentError) as excinfo:  # 验证下方操作会抛出缺少文本层异常。
        parse_pdf("scan.pdf", _blank_pdf())  # 解析。

    message = str(excinfo.value)  # 将消息设为值的字符串形式。
    assert "image-based" in message  # 确认“image-based”包含在消息。
    assert "OCR" in message  # 确认“OCR”包含在消息。


def test_low_density_pdf_is_rejected():  # 验证每页文本过少时拒绝入库。
    # 每页只写一小行，模拟扫描件上附带少量可提取的印章文字。
    data = _build_pdf(["x"] * 10)  # 将数据设为构造测试文档的结果。

    with pytest.raises(ScannedDocumentError):  # 验证下方操作会抛出缺少文本层异常。
        parse_pdf("sparse.pdf", data, min_chars_per_page=40)  # 解析。


def test_density_threshold_is_configurable():  # 验证文本密度阈值可配置。
    data = _build_pdf(["A short but real sentence of text, repeated to reach a plausible page length. " * 2] * 3)  # 将数据设为构造测试文档的结果。

    # 提高阈值，验证该文件被判为文本不足。
    with pytest.raises(ScannedDocumentError):  # 验证下方操作会抛出缺少文本层异常。
        parse_pdf("sample.pdf", data, min_chars_per_page=10_000)  # 解析。

    # 降低阈值，验证同一文件可以通过。
    assert not parse_pdf("sample.pdf", data, min_chars_per_page=1).is_empty()  # 确认不满足“为空的结果”。


def test_repeated_header_is_removed():  # 重复页眉位于顶部边缘，应排除以节省上下文预算。
    """重复页眉位于顶部边缘，应排除以节省上下文预算。"""
    pages = [f"Unique body text for page {i} with enough characters to matter." for i in range(5)]  # 将页面列表设为从指定的下标范围中提取的各项。

    document = parse_pdf("report.pdf", _build_pdf(pages, header="Running Header"))  # 将文档设为解析的结果。

    joined = " ".join(s.text for s in document.segments)  # 将拼接结果设为拼接后的从结构片段列表中提取的各项。
    assert "Running Header" not in joined  # 确认“Running Header”未包含在拼接结果。
    assert "Unique body text" in joined  # 确认“Unique body text”包含在拼接结果。


def test_body_text_repeating_on_every_page_is_kept():  # 模板正文也可能跨页重复。
    """模板正文也可能跨页重复；只有同时位于页边距的重复文本才应过滤。"""
    repeated = "This exact sentence appears on every page because the document is a template. "  # 设置repeated。
    pages = [repeated * 2 + f"Page specific value {i}." for i in range(5)]  # 将页面列表设为从指定的下标范围中提取的各项。

    document = parse_pdf("template.pdf", _build_pdf(pages))  # 将文档设为解析的结果。

    assert any("appears on every page" in s.text for s in document.segments)  # 确认至少一项满足“片段的文本包含“appears on every page””。


def test_page_numbers_are_dropped():  # 单独占一行的页码应在切块前过滤。
    """单独占一行的页码应在切块前过滤。"""
    body = "Real content that should survive the filter, long enough to look like a page. " * 3  # 将正文设为文本内容乘以3。
    data = _build_pdf([body + "\n2", body + "\n3", body + "\n4"])  # 将数据设为构造测试文档的结果。

    document = parse_pdf("doc.pdf", data)  # 将文档设为解析的结果。

    assert not any(s.text.strip().isdigit() for s in document.segments)  # 确认不满足“是否存在满足条件的项的结果”。


def test_corrupt_pdf_raises_value_error():  # 验证损坏文件触发值错误。
    with pytest.raises(ValueError):  # 验证下方操作会抛出值错误。
        parse_pdf("broken.pdf", b"not a pdf at all")  # 解析。


def test_removed_furniture_is_counted():  # 验证统计被过滤的页眉页脚。
    pages = [f"Body {i} with distinct wording that fills the page out properly." for i in range(4)]  # 将页面列表设为从指定的下标范围中提取的各项。

    document = parse_pdf("doc.pdf", _build_pdf(pages, header="Same Header"))  # 将文档设为解析的结果。

    assert document.metadata["removed_furniture"] >= 1  # 确认文档的元数据中的removed_furniture不小于1。


class TestHelpers:  # 定义 TestHelpers 类型。
    @pytest.mark.parametrize(  # 用多组输入重复验证同一行为。
        "line,expected",  # 为参数化用例指定原始行和期望判断两个参数。
        [  # 开始列出各项。
            ("12", True),  # 为parametrize提供输入。
            ("- 12 -", True),  # 为parametrize提供输入。
            ("iv", True),  # 为parametrize提供输入。
            ("Page 12 of 30", False),  # 为parametrize提供输入。
            ("2024", True),  # 为parametrize提供输入。
            ("a1", False),  # 为parametrize提供输入。
            ("", False),  # 为parametrize提供输入。
        ],  # 结束列表。
    )  # 结束当前多行表达式。
    def test_page_number_detection(self, line, expected):  # 验证识别不同写法的页码。
        assert _is_page_number(line) is expected  # 确认判断是否为页码的结果为预期值。

    def test_space_inserted_between_latin_words(self):  # 验证跨行拉丁单词间按需补空格。
        assert _needs_space("hello", "world") is True  # 确认判断行间是否需要空格的结果为真。

    def test_no_space_inserted_for_chinese(self):  # 验证中文换行处直接拼接。
        # 中文换行处直接拼合，避免在词中插入空格。
        assert _needs_space("这是第一句。", "这是第二句。") is False  # 确认判断行间是否需要空格的结果为假。

    def test_no_space_after_hyphenated_break(self):  # 验证断行连字符后无需插入空格。
        assert _needs_space("inter-", "national") is False  # 确认判断行间是否需要空格的结果为假。


def test_document_is_never_emptied_by_furniture_detection():  # 过滤逻辑处理后，有文本的文件仍应产生片段。
    """过滤逻辑处理后，有文本的文件仍应产生片段。"""
    pages = ["Header Text"] * 6  # 将页面列表设为列表乘以6。

    document = parse_pdf("headers.pdf", _build_pdf(pages), min_chars_per_page=1)  # 将文档设为解析的结果。

    assert not document.is_empty()  # 确认不满足“为空的结果”。
