"""验证文件类型识别和对应解析器的选择。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pymupdf  # 引入版面解析库。
import pytest  # 引入测试框架。

from unittest.mock import Mock

from rag_demo.config import Settings
from rag_demo.ingestion import IngestionError, IngestionPipeline
from rag_demo.parsers.base import UnsupportedDocumentError, check_extension, document_id, supported_extensions_display


@pytest.fixture
def pipeline():
    registry = Mock()
    registry.find_by_name.return_value = []
    embedder = Mock()
    embedder.embed_texts.side_effect = lambda texts: [[3.0, 4.0] for _ in texts]
    return IngestionPipeline(Settings(), Mock(), registry, embedder)


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


def test_markdown_dispatch(pipeline):
    pipeline.ingest("notes.md", b"# Title\n\nSome body text here.")
    children, parents, vectors = pipeline.store.insert_units.call_args.args
    assert children[0].breadcrumb == "Title"
    assert children[0].parent_id == parents[0].parent_id
    assert vectors == [[3.0, 4.0]]


@pytest.mark.parametrize("name", ["doc.pdf", "fake.md"])
def test_pdf_dispatch(pipeline, name):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Real text content that clears the density threshold comfortably here.")
        pipeline.ingest(name, pdf.tobytes())
    children, _, _ = pipeline.store.insert_units.call_args.args
    assert children[0].page == 1
    assert "Real text content" in children[0].text


def test_empty_content_is_rejected(pipeline):
    with pytest.raises(IngestionError, match="parsing: empty.md is empty"):
        pipeline.ingest("empty.md", b"")


def test_scanned_pdf_error_propagates_through_dispatch(pipeline):
    with pymupdf.open() as pdf:
        pdf.new_page()
        with pytest.raises(IngestionError, match="OCR"):
            pipeline.ingest("scan.pdf", pdf.tobytes())
    pipeline.embedder.embed_texts.assert_not_called()


def test_markdown_named_pdf_uses_pdf_parser(pipeline):
    with pytest.raises(IngestionError, match="PDF"):
        pipeline.ingest("real.pdf", b"# Heading")


def test_document_id_is_stable_and_content_sensitive():  # 验证相同文件标识稳定且内容改变会产生新标识。
    first = document_id("a.md", b"same")  # 将首项设为文档标识的结果。
    second = document_id("a.md", b"same")  # 将第二项设为文档标识的结果。
    changed = document_id("a.md", b"different")  # 将changed设为文档标识的结果。
    renamed = document_id("b.md", b"same")  # 将renamed设为文档标识的结果。

    assert first == second  # 确认首项等于第二项。
    assert first != changed  # 确认首项不等于changed。
    assert first != renamed  # 确认首项不等于renamed。