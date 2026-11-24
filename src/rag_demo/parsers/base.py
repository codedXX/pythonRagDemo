"""定义解析器共用的文档表示。各种格式统一输出结构化片段及来源信息，后续切块逻辑可独立于原文件格式。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import hashlib  # 引入摘要计算模块。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。
from enum import Enum  # 从 enum 导入枚举。
from pathlib import Path  # 从 pathlib 导入路径对象。


class SegmentKind(str, Enum):  # 定义 SegmentKind 类型。
    HEADING = "heading"  # 标记标题片段。
    PARAGRAPH = "paragraph"  # 标记普通正文段落。
    LIST = "list"  # 标记连续的列表项。
    CODE = "code"  # 标记围栏中的完整代码。
    TABLE = "table"  # 标记由多行组成的表格。
    QUOTE = "quote"  # 标记引用文字。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class StructuralSegment:  # 表示一段结构完整的文本及其来源。
    """表示一段结构完整的文本及其来源。标题深度和上层标题路径用于恢复段落位置。"""

    text: str  # 声明文本字段。
    kind: SegmentKind  # 声明类别字段。
    page: int = 0  # 设置页码字段的默认值。
    depth: int = 0  # 设置标题层级字段的默认值。
    heading_path: tuple[str, ...] = ()  # 设置标题路径字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def breadcrumb(self) -> str:  # 定义标题路径的处理逻辑。
        return " > ".join(self.heading_path)  # 用层级分隔符拼出来源的标题路径。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class ParsedDocument:  # 保存解析器得到的文档信息及有序片段。
    """保存解析器得到的文档信息及有序片段。"""

    name: str  # 声明名称字段。
    segments: list[StructuralSegment]  # 声明结构片段列表字段。
    page_count: int = 0  # 设置总页数字段的默认值。
    char_count: int = 0  # 设置字符数字段的默认值。
    metadata: dict = field(default_factory=dict)  # 设置元数据字段的默认值。

    def is_empty(self) -> bool:  # 判断当前内容是否为空。
        return not any(segment.text.strip() for segment in self.segments)  # 只要有一个片段含有非空白正文，文档就不为空。

    def chars_per_page(self) -> float:  # 定义每页平均字符数的处理逻辑。
        if self.page_count <= 0:  # 当总页数不超过0时进入此分支。
            return float("inf") if self.char_count == 0 else float(self.char_count)  # 没有有效页数时，按字符数选择无穷大或现有字符数，避免除零。
        return self.char_count / self.page_count  # 用总字符数除以页数计算平均文本密度。


class UnsupportedDocumentError(ValueError):  # 文件格式不受支持。
    """文件格式不受支持。"""


class ScannedDocumentError(ValueError):  # 文档没有足够的可提取文本，需要先进行文字识别。
    """文档没有足够的可提取文本，需要先进行文字识别。"""


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown"})  # 将允许的扩展名设为frozenset的结果。


def supported_extensions_display() -> str:  # 定义允许格式的提示文本的处理逻辑。
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))  # 把支持的扩展名排序后拼成提示文字。


def check_extension(name: str) -> str:  # 校验文件扩展名并转成小写。
    """校验文件扩展名并转成小写。"""
    suffix = Path(name).suffix.lower()  # 提取文件最后一个扩展名并统一为小写。
    if suffix not in SUPPORTED_EXTENSIONS:  # 当扩展名未包含在允许的扩展名时进入此分支。
        shown = suffix or "(no extension)"  # 没有扩展名时为错误提示提供可读的占位说明。
        raise UnsupportedDocumentError(  # 向上抛出不支持的文档格式异常。
            f"unsupported file type {shown}; supported types are "  # 列出被拒绝的扩展名并引出可用格式。
            f"{supported_extensions_display()}"  # 附上程序实际支持的扩展名列表。
        )  # 结束当前多行表达式。
    return suffix  # 返回扩展名。


def document_id(name: str, content: bytes) -> str:  # 根据文件名和内容计算稳定标识。
    """根据文件名和内容计算稳定标识；相同文件重复上传得到同一标识，内容改变则生成新标识。"""
    digest = hashlib.sha256()  # 初始化用于生成文档标识的摘要计算器。
    digest.update(name.encode("utf-8"))  # 把文件名的编码字节计入文档摘要。
    digest.update(b"\x00")  # 用零字节分隔文件名与正文，避免组合歧义。
    digest.update(content)  # 把原始文件内容计入摘要，使修改后的文件得到新标识。
    return digest.hexdigest()[:32]  # 取摘要的前三十二个十六进制字符作为文档标识。