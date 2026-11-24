"""识别上传文件的格式，并交给对应的解析器。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

from rag_demo.config import ParsingSettings  # 从 rag_demo.config 导入解析配置。
from rag_demo.parsers.base import (  # 从 rag_demo.parsers.base 引入本模块使用的类型和工具。
    ParsedDocument,  # 导入解析文档。
    ScannedDocumentError,  # 导入缺少文本层异常。
    UnsupportedDocumentError,  # 导入不支持的文档格式异常。
    check_extension,  # 导入校验扩展名。
    document_id,  # 导入文档标识。
)  # 结束当前多行表达式。
from rag_demo.parsers.markdown import MarkdownParser  # 从 rag_demo.parsers.markdown 导入标记文本解析器。
from rag_demo.parsers.pdf import PdfParser  # 从 rag_demo.parsers.pdf 导入便携文档解析器。

__all__ = [  # 将是否各项均满足条件设为列表。
    "ParsedDocument",  # 公开解析文档的统一表示。
    "ScannedDocumentError",  # 公开扫描件识别异常。
    "UnsupportedDocumentError",  # 公开不支持格式的异常。
    "check_extension",  # 公开扩展名校验函数。
    "document_id",  # 公开文档标识生成函数。
    "parse_document",  # 公开统一的文件解析入口。
]  # 结束列表。


def parse_document(name: str, content: bytes, settings: ParsingSettings) -> ParsedDocument:  # 按文件类型解析上传内容。
    """按文件类型解析上传内容；格式不受支持或扫描件缺少文本层时抛出对应异常。"""
    suffix = sniff_extension(name, content)  # 将扩展名设为识别真实文件类型的结果。

    if not content:  # 当内容为空或未启用时进入此分支。
        raise ValueError(f"{name} is empty")  # 向上抛出值错误。

    if suffix == ".pdf":  # 当扩展名等于“.pdf”时进入此分支。
        return PdfParser(min_chars_per_page=settings.min_chars_per_page).parse(name, content)  # 返回解析的结果。
    return MarkdownParser().parse(name, content)  # 返回解析的结果。


def sniff_extension(name: str, content: bytes) -> str:  # 结合扩展名和文件头识别类型，防止把改名的 PDF 当作文本读取。
    """结合扩展名和文件头识别类型，防止把改名的 PDF 当作文本读取。"""
    declared = check_extension(name)  # 将declared设为校验扩展名的结果。
    if content.startswith(b"%PDF-") and declared != ".pdf":  # 当前缀匹配的结果且declared不等于“.pdf”时进入此分支。
        return ".pdf"  # 返回“.pdf”。
    return declared  # 返回declared。