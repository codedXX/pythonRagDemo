"""PDF 和 Markdown 共用的文档类型与文件标识。"""

from rag_demo.parsers.base import (
    ParsedDocument,
    ScannedDocumentError,
    UnsupportedDocumentError,
    check_extension,
    document_id,
)

__all__ = [
    "ParsedDocument",
    "ScannedDocumentError",
    "UnsupportedDocumentError",
    "check_extension",
    "document_id",
]
