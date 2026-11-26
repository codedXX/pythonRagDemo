"""PDF 和 Markdown 共用的文档类型与文件标识。"""

from rag_demo.parsers.base import (  # 集中引入供上层调用的解析接口。
    ParsedDocument,  # 统一表示解析后的文档。
    ScannedDocumentError,  # 区分需要 OCR 的 PDF 解析错误。
    UnsupportedDocumentError,  # 区分不支持的文件类型。
    check_extension,  # 提供文件扩展名校验函数。
    document_id,  # 提供文档标识生成函数。
)  # 结束解析接口导入。

__all__ = [  # 声明解析包对外公开的名称。
    "ParsedDocument",  # 导出文档结果类型。
    "ScannedDocumentError",  # 导出扫描件异常。
    "UnsupportedDocumentError",  # 导出文件类型异常。
    "check_extension",  # 导出扩展名校验函数。
    "document_id",  # 导出文档标识生成函数。
]  # 完成公开接口清单。
