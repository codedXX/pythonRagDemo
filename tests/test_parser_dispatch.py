"""验证文件类型识别和对应解析器的选择。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pymupdf  # 引入版面解析库。
import pytest  # 引入测试框架。

from rag_demo.config import ParsingSettings  # 从 rag_demo.config 导入解析配置。
from rag_demo.parsers import (  # 从 rag_demo.parsers 引入本模块使用的类型和工具。
    ScannedDocumentError,  # 导入缺少文本层异常。
    UnsupportedDocumentError,  # 导入不支持的文档格式异常。
    check_extension,  # 导入校验扩展名。
    document_id,  # 导入文档标识。
    parse_document,  # 导入解析文档。
    sniff_extension,  # 导入识别真实文件类型。
)  # 结束当前多行表达式。
from rag_demo.parsers.base import supported_extensions_display  # 从 rag_demo.parsers.base 导入允许格式的提示文本。


@pytest.mark.parametrize("name", ["a.pdf", "b.md", "c.markdown", "D.PDF", "e.Md"])  # 用多组输入重复验证同一行为。
def test_supported_extensions_are_accepted(name):  # 验证接受已支持的扩展名。
    assert check_extension(name) in {".pdf", ".md", ".markdown"}  # 确认校验扩展名的结果包含在集合。


@pytest.mark.parametrize("name", ["a.txt", "b.docx", "c.html", "d", "e.xlsx"])  # 用多组输入重复验证同一行为。
def test_unsupported_extensions_are_rejected(name):  # 验证拒绝未知扩展名。
    with pytest.raises(UnsupportedDocumentError):  # 验证下方操作会抛出不支持的文档格式异常。
        check_extension(name)  # 校验扩展名。


def test_rejection_message_lists_supported_types():  # 验证拒绝提示列出允许的格式。
    with pytest.raises(UnsupportedDocumentError) as excinfo:  # 验证下方操作会抛出不支持的文档格式异常。
        check_extension("notes.txt")  # 校验扩展名。

    message = str(excinfo.value)  # 将消息设为值的字符串形式。
    assert ".txt" in message  # 确认“.txt”包含在消息。
    assert supported_extensions_display() == ".markdown, .md, .pdf"  # 确认允许格式的提示文本的结果等于“.markdown, .md, .pdf”。


def test_markdown_dispatch():  # 验证标记文档交给对应解析器。
    document = parse_document(  # 将文档设为解析文档的结果。
        "notes.md", "# Title\n\nSome body text here.\n".encode(), ParsingSettings()  # 传入带标题的文本文件和默认解析配置。
    )  # 结束当前多行表达式。

    assert document.metadata["format"] == "markdown"  # 确认文档的元数据中的format等于“markdown”。
    assert not document.is_empty()  # 确认不满足“为空的结果”。


def test_pdf_dispatch():  # 验证便携文档交给对应解析器。
    pdf = pymupdf.open()  # 将便携文档设为打开文档的结果。
    page = pdf.new_page()  # 将页码设为创建页面的结果。
    page.insert_textbox(  # 写入文本框。
        pymupdf.Rect(72, 72, 523, 770),  # 为Rect提供输入。
        "A page of real text that is long enough to clear the density threshold.",  # 提供足以通过密度阈值的真实文本。
        fontsize=11,  # 传入fontsize参数。
    )  # 结束当前多行表达式。
    content = pdf.tobytes()  # 将内容设为导出文档字节串的结果。
    pdf.close()  # 关闭便携文档并释放资源。

    document = parse_document("doc.pdf", content, ParsingSettings())  # 将文档设为解析文档的结果。

    assert document.metadata["format"] == "pdf"  # 确认文档的元数据中的format等于“pdf”。


def test_empty_content_is_rejected():  # 验证拒绝空文件内容。
    with pytest.raises(ValueError):  # 验证下方操作会抛出值错误。
        parse_document("empty.md", b"", ParsingSettings())  # 解析文档。


def test_scanned_pdf_error_propagates_through_dispatch():  # 验证扫描件错误能传递给调用方。
    pdf = pymupdf.open()  # 将便携文档设为打开文档的结果。
    pdf.new_page()  # 创建页面。
    content = pdf.tobytes()  # 将内容设为导出文档字节串的结果。
    pdf.close()  # 关闭便携文档并释放资源。

    with pytest.raises(ScannedDocumentError):  # 验证下方操作会抛出缺少文本层异常。
        parse_document("scan.pdf", content, ParsingSettings())  # 解析文档。


def test_pdf_renamed_to_markdown_is_still_treated_as_pdf():  # 文件头优先于扩展名，以免将二进制内容误读为文字。
    """文件头优先于扩展名，以免将二进制内容误读为文字。"""
    pdf = pymupdf.open()  # 将便携文档设为打开文档的结果。
    page = pdf.new_page()  # 将页码设为创建页面的结果。
    page.insert_textbox(  # 写入文本框。
        pymupdf.Rect(72, 72, 523, 770),  # 为Rect提供输入。
        "Real text content that clears the density threshold comfortably here.",  # 构造改名后仍能作为有效便携文档解析的正文。
        fontsize=11,  # 传入fontsize参数。
    )  # 结束当前多行表达式。
    content = pdf.tobytes()  # 将内容设为导出文档字节串的结果。
    pdf.close()  # 关闭便携文档并释放资源。

    assert sniff_extension("fake.md", content) == ".pdf"  # 确认识别真实文件类型的结果等于“.pdf”。
    assert parse_document("fake.md", content, ParsingSettings()).metadata["format"] == "pdf"  # 确认元数据中的format等于“pdf”。


def test_markdown_named_pdf_stays_markdown():  # 验证带便携文档后缀的标记文本保持文本类型。
    assert sniff_extension("real.pdf", b"# Heading\n\ntext\n") == ".pdf"  # 确认识别真实文件类型的结果等于“.pdf”。


def test_document_id_is_stable_and_content_sensitive():  # 验证相同文件标识稳定且内容改变会产生新标识。
    first = document_id("a.md", b"same")  # 将首项设为文档标识的结果。
    second = document_id("a.md", b"same")  # 将第二项设为文档标识的结果。
    changed = document_id("a.md", b"different")  # 将changed设为文档标识的结果。
    renamed = document_id("b.md", b"same")  # 将renamed设为文档标识的结果。

    assert first == second  # 确认首项等于第二项。
    assert first != changed  # 确认首项不等于changed。
    assert first != renamed  # 确认首项不等于renamed。