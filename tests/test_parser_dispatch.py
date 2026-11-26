"""验证文件类型识别和对应解析器的选择。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pymupdf  # 引入版面解析库。
import pytest  # 引入测试框架。

from unittest.mock import Mock  # 用替身记录存储调用并固定向量服务响应。

from rag_demo.config import Settings  # 使用默认项目配置构造入库流程。
from rag_demo.ingestion import IngestionError, IngestionPipeline  # 引入真实入库流程及其对外异常。
from rag_demo.parsers.base import UnsupportedDocumentError, check_extension, document_id, supported_extensions_display  # 引入扩展名校验、文档标识及支持格式的说明接口。


@pytest.fixture  # 注册多个解析分派用例共享的夹具。
def pipeline():  # 为每个用例组装隔离的入库流程。
    registry = Mock()  # 用替身接收文档状态更新。
    registry.find_by_name.return_value = []  # 模拟当前没有需要替换的同名文档。
    embedder = Mock()  # 构造向量服务替身。
    embedder.embed_texts.side_effect = lambda texts: [[3.0, 4.0] for _ in texts]  # 为每个输入文本返回固定二维向量。
    return IngestionPipeline(Settings(), Mock(), registry, embedder)  # 组装真实入库逻辑与存储、编码替身。


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


def test_markdown_dispatch(pipeline):  # 验证 Markdown 文件进入对应解析流程。
    pipeline.ingest("notes.md", b"# Title\n\nSome body text here.")  # 导入含标题和正文的 Markdown 示例。
    children, parents, vectors = pipeline.store.insert_units.call_args.args  # 取出实际写入的子块、父块和向量。
    assert children[0].breadcrumb == "Title"  # 核对 Markdown 标题传递到了子块的标题路径。
    assert children[0].parent_id == parents[0].parent_id  # 确认子块关联到本次生成的父块。
    assert vectors == [[3.0, 4.0]]  # 确认编码结果与子块一起提交给存储。


@pytest.mark.parametrize("name", ["doc.pdf", "fake.md"])  # 分别测试真实 PDF 后缀和误用 Markdown 后缀的文件。
def test_pdf_dispatch(pipeline, name):  # 验证 PDF 内容按页解析并保留出处。
    with pymupdf.open() as pdf:  # 在内存中构造真正的 PDF。
        page = pdf.new_page()  # 添加承载示例正文的第一页。
        page.insert_text((72, 72), "Real text content that clears the density threshold comfortably here.")  # 写入足够长的文本以通过扫描件密度检查。
        pipeline.ingest(name, pdf.tobytes())  # 按指定文件名导入同一份 PDF 字节。
    children, _, _ = pipeline.store.insert_units.call_args.args  # 从存储调用中取得解析和分块后的子块。
    assert children[0].page == 1  # 确认来源页码从一开始。
    assert "Real text content" in children[0].text  # 确认实际提取出了 PDF 中的文字。


def test_empty_content_is_rejected(pipeline):  # 验证空文件会在解析阶段失败。
    with pytest.raises(IngestionError, match="parsing: empty.md is empty"):  # 匹配带文件名与阶段标记的错误信息。
        pipeline.ingest("empty.md", b"")  # 将空字节内容交给入库入口。


def test_scanned_pdf_error_propagates_through_dispatch(pipeline):  # 验证扫描件错误能通过入库流程传递出去。
    with pymupdf.open() as pdf:  # 创建没有文本层内容的 PDF 样本。
        pdf.new_page()  # 加入一张空白页以触发密度检查。
        with pytest.raises(IngestionError, match="OCR"):  # 期待上游仍能看到需要 OCR 的提示。
            pipeline.ingest("scan.pdf", pdf.tobytes())  # 尝试将空白 PDF 导入知识库。
    pipeline.embedder.embed_texts.assert_not_called()  # 确认解析失败后没有调用向量服务。


def test_markdown_named_pdf_uses_pdf_parser(pipeline):  # 验证 PDF 后缀会让 Markdown 文本进入 PDF 解析器。
    with pytest.raises(IngestionError, match="PDF"):  # 期待错误说明提及 PDF 解析。
        pipeline.ingest("real.pdf", b"# Heading")  # 用 PDF 文件名提交 Markdown 字节以触发解析失败。


def test_document_id_is_stable_and_content_sensitive():  # 验证相同文件标识稳定且内容改变会产生新标识。
    first = document_id("a.md", b"same")  # 将首项设为文档标识的结果。
    second = document_id("a.md", b"same")  # 将第二项设为文档标识的结果。
    changed = document_id("a.md", b"different")  # 将changed设为文档标识的结果。
    renamed = document_id("b.md", b"same")  # 将renamed设为文档标识的结果。

    assert first == second  # 确认首项等于第二项。
    assert first != changed  # 确认首项不等于changed。
    assert first != renamed  # 确认首项不等于renamed。