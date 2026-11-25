"""本机 MySQL 文档登记库，片段正文由 Milvus 保存。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import pymysql
from pymysql.cursors import DictCursor

from rag_demo.config import MySQLSettings

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


@contextmanager
def registry_transaction(settings: MySQLSettings) -> Iterator[DictCursor]:
    with pymysql.connect(**settings.model_dump(), charset="utf8mb4",
                         cursorclass=DictCursor, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            try:
                yield cursor
                connection.commit()
            except Exception:
                connection.rollback()
                raise


class DocumentRegistry:
    def __init__(self, settings: MySQLSettings) -> None:
        self.settings = settings
        try:
            with registry_transaction(settings) as cursor:
                cursor.execute(_SCHEMA)
        except pymysql.MySQLError as exc:
            raise RuntimeError(
                f"MySQL registry {settings.host}:{settings.port}/{settings.database}: {exc}"
            ) from exc

    def begin(self, doc_id: str, name: str) -> DocumentRecord:
        with registry_transaction(self.settings) as cursor:
            cursor.execute(
                """
                INSERT INTO documents (doc_id, name, state, created_at, updated_at)
                VALUES (%s, %s, 'staging', UTC_TIMESTAMP(), UTC_TIMESTAMP())
                ON DUPLICATE KEY UPDATE name = %s, state = 'staging',
                    parent_count = 0, child_count = 0, char_count = 0,
                    updated_at = UTC_TIMESTAMP(), error = NULL
                """,
                (doc_id, name, name),
            )
            cursor.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
            return _to_record(cursor.fetchone())

    def mark_active(self, doc_id: str, parent_count: int, child_count: int, char_count: int) -> None:
        with registry_transaction(self.settings) as cursor:
            cursor.execute(
                """
                UPDATE documents SET state = 'active', parent_count = %s, child_count = %s,
                    char_count = %s, updated_at = UTC_TIMESTAMP(), error = NULL
                WHERE doc_id = %s
                """,
                (parent_count, child_count, char_count, doc_id),
            )

    def mark_failed(self, doc_id: str, error: str) -> None:
        with registry_transaction(self.settings) as cursor:
            cursor.execute(
                """UPDATE documents SET state = 'failed', error = %s, updated_at = UTC_TIMESTAMP()
                   WHERE doc_id = %s""",
                (error[:2000], doc_id),
            )

    def remove(self, doc_id: str) -> None:
        with registry_transaction(self.settings) as cursor:
            cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def _records(self, query: str, args: tuple = ()) -> list[DocumentRecord]:
        with registry_transaction(self.settings) as cursor:
            cursor.execute(query, args)
            return [_to_record(row) for row in cursor.fetchall()]

    def get(self, doc_id: str) -> DocumentRecord | None:
        rows = self._records("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
        return rows[0] if rows else None

    def find_by_name(self, name: str) -> list[DocumentRecord]:
        return self._records("SELECT * FROM documents WHERE name = %s ORDER BY created_at", (name,))

    def list_documents(self) -> list[DocumentRecord]:
        return self._records("SELECT * FROM documents ORDER BY created_at DESC, name")

    def list_usable(self) -> list[DocumentRecord]:
        return self._records("SELECT * FROM documents WHERE state = 'active' ORDER BY created_at DESC")

    def counts(self) -> dict[str, int]:
        with registry_transaction(self.settings) as cursor:
            cursor.execute("SELECT state, COUNT(*) AS n FROM documents GROUP BY state")
            return {row["state"]: row["n"] for row in cursor.fetchall()}


def _to_record(row: dict) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        name=row["name"],
        state=DocumentState(row["state"]),
        parent_count=row["parent_count"],
        child_count=row["child_count"],
        char_count=row["char_count"],
        created_at=row["created_at"].isoformat(sep=" "),
        updated_at=row["updated_at"].isoformat(sep=" "),
        error=row["error"],
    )
