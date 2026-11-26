"""本机 MySQL 文档登记库，片段正文由 Milvus 保存。"""

from __future__ import annotations  # 将类型注解的求值延后。

from collections.abc import Iterator  # 声明事务上下文交付的游标类型。
from contextlib import contextmanager  # 用生成器管理事务的进入和退出。
from dataclasses import dataclass  # 将登记字段组织为数据类。
from enum import Enum  # 用枚举表示文档状态。

import pymysql  # 使用 PyMySQL 连接登记数据库。
from pymysql.cursors import DictCursor  # 使查询结果可按列名读取。

from rag_demo.config import MySQLSettings  # 从项目配置取得登记库连接参数。

# documents 以 doc_id 为主键，分别保存导入状态、块数、字符数、时间戳和失败原因。
# 文件名索引取前 191 个字符，状态索引用于筛选可用文档。二进制排序规则区分大小写。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    name LONGTEXT NOT NULL,
    state VARCHAR(16) NOT NULL,
    parent_count BIGINT NOT NULL DEFAULT 0,
    child_count BIGINT NOT NULL DEFAULT 0,
    char_count BIGINT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    error LONGTEXT,
    INDEX idx_documents_name (name(191)),
    INDEX idx_documents_state (state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin
"""


class DocumentState(str, Enum):  # 定义 DocumentState 类型。
    STAGING = "staging"  # 声明“入库中”状态的序列化值。
    ACTIVE = "active"  # 声明“已激活”状态的序列化值。
    FAILED = "failed"  # 声明“失败”状态的序列化值。


@dataclass(slots=True)  # 根据字段声明生成数据类的初始化和比较方法。
class DocumentRecord:  # 保存单份文档的登记信息。
    """保存单份文档的登记信息。"""

    doc_id: str  # 声明文档标识字段。
    name: str  # 声明名称字段。
    state: DocumentState  # 声明状态字段。
    parent_count: int = 0  # 设置父块数字段的默认值。
    child_count: int = 0  # 设置子块数字段的默认值。
    char_count: int = 0  # 设置字符数字段的默认值。
    created_at: str = ""  # 设置创建时间字段的默认值。
    updated_at: str = ""  # 设置更新时间字段的默认值。
    error: str | None = None  # 设置错误字段的默认值。

    @property  # 允许调用方像读取属性一样取得计算结果。
    def usable(self) -> bool:  # 判断文档是否已激活并可用于检索。
        return self.state is DocumentState.ACTIVE  # 只有激活状态的文档才可用于问答。

    def as_dict(self) -> dict:  # 将对象字段转换为可序列化的字典。
        return {  # 返回字段映射。
            "doc_id": self.doc_id,  # 填入文档标识字段。
            "name": self.name,  # 填入名称字段。
            "state": self.state.value,  # 填入状态字段。
            "parent_count": self.parent_count,  # 填入父块数字段。
            "child_count": self.child_count,  # 填入子块数字段。
            "char_count": self.char_count,  # 填入字符数字段。
            "created_at": self.created_at,  # 填入创建时间字段。
            "updated_at": self.updated_at,  # 填入更新时间字段。
            "error": self.error,  # 填入错误字段。
        }  # 结束本组字段。


@contextmanager  # 把事务生成器包装成可用于 with 的上下文。
def registry_transaction(settings: MySQLSettings) -> Iterator[DictCursor]:  # 让每次登记操作在同一连接内提交或回滚。
    with pymysql.connect(**settings.model_dump(), charset="utf8mb4",  # 使用项目配置连接 MySQL 并支持完整 Unicode 字符。
                         cursorclass=DictCursor, connect_timeout=5) as connection:  # 用字典游标读取结果并将连接超时设为五秒。
        with connection.cursor() as cursor:  # 在本次事务范围内打开并释放游标。
            try:  # 在调用方执行数据库语句期间跟踪异常。
                yield cursor  # 把同一事务的游标借给调用方。
                connection.commit()  # 调用方正常结束后提交全部修改。
            except Exception:  # 捕获操作或提交期间抛出的异常。
                connection.rollback()  # 撤销本次事务中的未提交修改。
                raise  # 向上游保留原始异常。


class DocumentRegistry:  # 集中管理文档状态和入库统计。
    def __init__(self, settings: MySQLSettings) -> None:  # 初始化登记库并确保登记表存在。
        self.settings = settings  # 保留供后续各次连接使用的配置。
        try:  # 将建表期间的数据库错误转为初始化错误。
            with registry_transaction(settings) as cursor:  # 在目标数据库上打开事务游标。
                cursor.execute(_SCHEMA)  # 执行预定义的建表语句。
        except pymysql.MySQLError as exc:  # 捕获 MySQL 连接或建表错误。
            raise RuntimeError(  # 对外报告登记库初始化失败。
                f"MySQL registry {settings.host}:{settings.port}/{settings.database}: {exc}"  # 列出主机、端口和库名以定位目标实例。
            ) from exc  # 保留底层驱动异常的原因链。

    def begin(self, doc_id: str, name: str) -> DocumentRecord:  # 新建或重置一条正在导入的文档记录。
        with registry_transaction(self.settings) as cursor:  # 让登记和随后的回读处于同一事务。
            cursor.execute(  # 插入新记录，主键重复时重置状态和统计。
                """
                INSERT INTO documents (doc_id, name, state, created_at, updated_at)
                VALUES (%s, %s, 'staging', UTC_TIMESTAMP(), UTC_TIMESTAMP())
                ON DUPLICATE KEY UPDATE name = %s, state = 'staging',
                    parent_count = 0, child_count = 0, char_count = 0,
                    updated_at = UTC_TIMESTAMP(), error = NULL
                """,
                (doc_id, name, name),  # 绑定主键以及插入和更新所需的文件名。
            )  # 完成正在导入状态的写入。
            cursor.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))  # 按主键回读包含数据库时间戳的记录。
            return _to_record(cursor.fetchone())  # 将回读结果转换成领域对象。

    def mark_active(self, doc_id: str, parent_count: int, child_count: int, char_count: int) -> None:  # 登记导入成功后的块数和字符数。
        with registry_transaction(self.settings) as cursor:  # 用独立事务发布登记状态。
            cursor.execute(  # 将目标记录更新为可用并清除旧错误。
                """
                UPDATE documents SET state = 'active', parent_count = %s, child_count = %s,
                    char_count = %s, updated_at = UTC_TIMESTAMP(), error = NULL
                WHERE doc_id = %s
                """,
                (parent_count, child_count, char_count, doc_id),  # 绑定三项最终统计及要更新的文档标识。
            )  # 完成激活状态与统计的更新。

    def mark_failed(self, doc_id: str, error: str) -> None:  # 保存导入失败状态和诊断信息。
        with registry_transaction(self.settings) as cursor:  # 在事务中更新目标文档。
            cursor.execute(  # 同时写入失败状态、错误和更新时间。
                """UPDATE documents SET state = 'failed', error = %s, updated_at = UTC_TIMESTAMP()
                   WHERE doc_id = %s""",
                (error[:2000], doc_id),  # 将错误说明限制为两千字符并指定目标记录。
            )  # 完成失败状态更新。

    def remove(self, doc_id: str) -> None:  # 按标识移除文档登记记录。
        with registry_transaction(self.settings) as cursor:  # 确保删除操作在事务退出时提交。
            cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))  # 用绑定参数删除指定的文档行。

    def _records(self, query: str, args: tuple = ()) -> list[DocumentRecord]:  # 执行查询并统一转换为文档记录列表。
        with registry_transaction(self.settings) as cursor:  # 在一个事务内读取本次查询的全部结果。
            cursor.execute(query, args)  # 绑定查询参数并执行 SQL。
            return [_to_record(row) for row in cursor.fetchall()]  # 逐行将数据库记录转换为应用对象。

    def get(self, doc_id: str) -> DocumentRecord | None:  # 按文档标识读取单条登记信息。
        rows = self._records("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))  # 按绑定的主键查询匹配记录。
        return rows[0] if rows else None  # 找到记录时返回首行，否则表示不存在。

    def find_by_name(self, name: str) -> list[DocumentRecord]:  # 查找所有同名文档以支持版本替换。
        return self._records("SELECT * FROM documents WHERE name = %s ORDER BY created_at", (name,))  # 按创建时间从早到晚返回同名记录。

    def list_documents(self) -> list[DocumentRecord]:  # 读取供界面展示的全部文档。
        return self._records("SELECT * FROM documents ORDER BY created_at DESC, name")  # 按创建时间倒序排列，同一时间再按名称排序。

    def list_usable(self) -> list[DocumentRecord]:  # 只列出可参与检索的文档。
        return self._records("SELECT * FROM documents WHERE state = 'active' ORDER BY created_at DESC")  # 筛选已激活文档并优先返回较新的记录。

    def counts(self) -> dict[str, int]:  # 统计各导入状态对应的文档数量。
        with registry_transaction(self.settings) as cursor:  # 在同一查询中取得各状态的计数。
            cursor.execute("SELECT state, COUNT(*) AS n FROM documents GROUP BY state")  # 按状态分组并由数据库计数。
            return {row["state"]: row["n"] for row in cursor.fetchall()}  # 以状态值为键返回数量。


def _to_record(row: dict) -> DocumentRecord:  # 将字典游标返回的数据库行转换为文档对象。
    return DocumentRecord(  # 构造使用项目字段类型的登记记录。
        doc_id=row["doc_id"],  # 保留文档的唯一标识。
        name=row["name"],  # 保留原始文档名称。
        state=DocumentState(row["state"]),  # 把数据库中的状态值还原为枚举。
        parent_count=row["parent_count"],  # 读取父块数。
        child_count=row["child_count"],  # 读取子块数。
        char_count=row["char_count"],  # 读取正文字符数。
        created_at=row["created_at"].isoformat(sep=" "),  # 将创建时间转为使用空格分隔日期和时间的文本。
        updated_at=row["updated_at"].isoformat(sep=" "),  # 以相同的格式输出最近更新时间。
        error=row["error"],  # 保留失败原因或数据库中的空值。
    )  # 完成登记记录转换。
