"""通过 SQLite 跟踪文档生命周期及失败原因，片段正文统一存放在 Milvus。

文档先登记为 staging，全部片段写入并激活后转为 active；中途失败则转为 failed。删除时同时移除登记行和对应内容。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import sqlite3  # 引入本地数据库驱动。
import threading  # 引入线程工具。
from dataclasses import dataclass  # 从 dataclasses 导入数据类装饰器。
from enum import Enum  # 从 enum 导入枚举。
from pathlib import Path  # 从 pathlib 导入路径对象。

# 初始化文档登记表，并为文件名和状态创建查询索引。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    state        TEXT NOT NULL,
    parent_count INTEGER NOT NULL DEFAULT 0,
    child_count  INTEGER NOT NULL DEFAULT 0,
    char_count   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(name);
CREATE INDEX IF NOT EXISTS idx_documents_state ON documents(state);
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


class DocumentRegistry:  # 用锁保护登记表操作，每次操作独立建立连接，供入库工作线程安全使用。
    """用锁保护登记表操作，每次操作独立建立连接，供入库工作线程安全使用。"""

    def __init__(self, path: Path | str) -> None:  # 保存实例所需的配置和依赖。
        self._path = Path(path)  # 将缓存文件路径设为路径对象的结果。
        self._path.parent.mkdir(parents=True, exist_ok=True)  # 确保数据库的父目录已经创建。
        self._lock = threading.Lock()  # 将互斥锁设为Lock的结果。
        with self._connect() as connection:  # 用独立连接初始化登记库。
            connection.executescript(_SCHEMA)  # 执行建表和建索引语句，已有结构可继续使用。

    def _connect(self) -> sqlite3.Connection:  # 定义_connect的处理逻辑。
        connection = sqlite3.connect(self._path, timeout=30.0)  # 将连接设为connect的结果。
        connection.row_factory = sqlite3.Row  # 让查询结果可以按列名读取。
        return connection  # 返回连接。

    # 登记状态变更。

    def begin(self, doc_id: str, name: str) -> DocumentRecord:  # 登记开始入库的文档。
        """登记开始入库的文档。"""
        with self._lock, self._connect() as connection:  # 持锁进行本次登记写入，由连接上下文管理事务。
            connection.execute(  # 通过数据库连接执行这条语句。
                # 按文档标识新增或重置登记记录，清空旧统计与错误原因。
                """
                INSERT INTO documents (doc_id, name, state)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    name = excluded.name,
                    state = excluded.state,
                    parent_count = 0,
                    child_count = 0,
                    char_count = 0,
                    updated_at = datetime('now'),
                    error = NULL
                """,
                (doc_id, name, DocumentState.STAGING.value),  # 按占位符顺序绑定文档标识、文件名和入库中状态。
            )  # 结束当前多行表达式。
        record = self.get(doc_id)  # 将登记记录设为当前实例中的对应值。
        assert record is not None  # 刚写入的登记记录应当能从数据库读回。
        return record  # 返回登记记录。

    def mark_active(  # 记录文档已成功入库及其片段数量。
        self,  # 通过实例读取本方法所需状态。
        doc_id: str,  # 接收文档标识参数。
        parent_count: int,  # 接收父块数参数。
        child_count: int,  # 接收子块数参数。
        char_count: int,  # 接收字符数参数。
    ) -> None:  # 结束参数声明，进入函数或语句主体。
        """记录文档已成功入库及其片段数量。"""
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            connection.execute(  # 通过数据库连接执行这条语句。
                # 将文档标为已激活，写入切块及字符统计并清除旧错误。
                """
                UPDATE documents
                   SET state = ?, parent_count = ?, child_count = ?,
                       char_count = ?, updated_at = datetime('now'), error = NULL
                 WHERE doc_id = ?
                """,
                (  # 开始组合多行参数。
                    DocumentState.ACTIVE.value,  # 传入DocumentState.ACTIVE.value。
                    parent_count,  # 传入父块数。
                    child_count,  # 传入子块数。
                    char_count,  # 传入字符数。
                    doc_id,  # 传入文档标识。
                ),  # 结束当前多行表达式。
            )  # 结束当前多行表达式。

    def mark_failed(self, doc_id: str, error: str) -> None:  # 记录失败状态和具体原因。
        """记录失败状态和具体原因。"""
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            connection.execute(  # 通过数据库连接执行这条语句。
                # 将文档标为失败，同时保存失败原因和更新时间。
                """
                UPDATE documents
                   SET state = ?, error = ?, updated_at = datetime('now')
                 WHERE doc_id = ?
                """,
                (DocumentState.FAILED.value, error[:2000], doc_id),  # 为execute提供输入。
            )  # 结束当前多行表达式。

    def remove(self, doc_id: str) -> None:  # 定义移除的处理逻辑。
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))  # 通过数据库连接执行这条语句。

    # 查询登记信息。

    def get(self, doc_id: str) -> DocumentRecord | None:  # 定义读取的处理逻辑。
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            row = connection.execute(  # 将记录设为fetchone的结果。
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)  # 按绑定的文档标识查询登记记录。
            ).fetchone()  # 为execute提供输入。
        return _to_record(row) if row else None  # 返回按条件选取的值。

    def find_by_name(self, name: str) -> list[DocumentRecord]:  # 定义按名称查询的处理逻辑。
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            rows = connection.execute(  # 将记录列表设为fetchall的结果。
                "SELECT * FROM documents WHERE name = ? ORDER BY created_at", (name,)  # 按文件名查找同名记录并按创建时间排序。
            ).fetchall()  # 为execute提供输入。
        return [_to_record(row) for row in rows]  # 返回从记录列表中提取的各项。

    def list_documents(self) -> list[DocumentRecord]:  # 按时间从新到旧列出所有文档，保留失败记录供界面解释原因。
        """按时间从新到旧列出所有文档，保留失败记录供界面解释原因。"""
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            rows = connection.execute(  # 将记录列表设为fetchall的结果。
                "SELECT * FROM documents ORDER BY created_at DESC, name"  # 按创建时间降序列出文档，同一时刻按名称排序。
            ).fetchall()  # 为execute提供输入。
        return [_to_record(row) for row in rows]  # 返回从记录列表中提取的各项。

    def list_usable(self) -> list[DocumentRecord]:  # 定义列出有效文档的处理逻辑。
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            rows = connection.execute(  # 将记录列表设为fetchall的结果。
                "SELECT * FROM documents WHERE state = ? ORDER BY created_at DESC",  # 只查询指定状态的文档，最新的排在前面。
                (DocumentState.ACTIVE.value,),  # 为execute提供输入。
            ).fetchall()  # 为execute提供输入。
        return [_to_record(row) for row in rows]  # 返回从记录列表中提取的各项。

    def counts(self) -> dict[str, int]:  # 定义按状态计数的处理逻辑。
        with self._lock, self._connect() as connection:  # 在互斥锁的作用域内执行后续操作。
            rows = connection.execute(  # 将记录列表设为fetchall的结果。
                "SELECT state, COUNT(*) AS n FROM documents GROUP BY state"  # 按状态分组统计文档数。
            ).fetchall()  # 为execute提供输入。
        return {row["state"]: row["n"] for row in rows}  # 返回从记录列表中提取的映射。


def _to_record(row: sqlite3.Row) -> DocumentRecord:  # 定义构造文档登记记录的处理逻辑。
    return DocumentRecord(  # 返回文档登记记录的结果。
        doc_id=row["doc_id"],  # 传入文档标识参数。
        name=row["name"],  # 传入名称参数。
        state=DocumentState(row["state"]),  # 传入状态参数。
        parent_count=row["parent_count"],  # 传入父块数参数。
        child_count=row["child_count"],  # 传入子块数参数。
        char_count=row["char_count"],  # 传入字符数参数。
        created_at=row["created_at"],  # 传入创建时间参数。
        updated_at=row["updated_at"],  # 传入更新时间参数。
        error=row["error"],  # 传入错误参数。
    )  # 结束当前多行表达式。