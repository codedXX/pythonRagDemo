"""停写后将 SQLite 文档登记表导入配置的本地 MySQL。"""

import argparse  # 解析迁移命令的参数。
import sqlite3  # 读取旧登记库中的 SQLite 记录。
import sys  # 将失败原因输出到标准错误。
from collections import Counter  # 按文档状态汇总迁移数量。
from contextlib import closing  # 退出读取范围时关闭 SQLite 连接。
from pathlib import Path  # 接收源数据库和配置文件的路径。

import pymysql  # 捕获 MySQL 驱动报告的迁移错误。

from rag_demo.config import ConfigurationError, MySQLSettings, load_settings  # 载入目标 MySQL 配置并识别配置错误。
from rag_demo.store.registry import DocumentRegistry, registry_transaction  # 复用登记表建表逻辑和事务管理。

COLUMNS = ("doc_id", "name", "state", "parent_count", "child_count", "char_count",  # 固定文档标识、名称、状态和统计字段的读写顺序。
           "created_at", "updated_at", "error")  # 将创建时间、更新时间和失败原因也纳入核对。
SELECT_ROWS = "SELECT " + ", ".join(COLUMNS) + " FROM documents"  # 用同一组列读取源表与目标表。


def target_rows(cursor) -> dict:  # 读取 MySQL 端的记录以便逐条核对。
    cursor.execute(SELECT_ROWS)  # 查询登记表里所有待比较的字段。
    rows = {}  # 按文档标识建立目标记录映射。
    for row in cursor.fetchall():  # 遍历查询返回的所有目标记录。
        for key in ("created_at", "updated_at"):  # 统一处理两个时间字段。
            row[key] = row[key].isoformat(sep=" ")  # 把 MySQL 时间转换为与 SQLite 可直接比较的文本。
        rows[row["doc_id"]] = row  # 保存整条目标记录供按标识查询。
    return rows  # 返回按标识整理的目标快照。


def verify_rows(source: dict, target: dict) -> None:  # 逐个核对两端的标识和字段。
    missing = source.keys() - target.keys()  # 找出尚未迁入目标库的标识。
    extra = target.keys() - source.keys()  # 找出仅存在于目标库的标识。
    if missing or extra:  # 两端标识集合有差异时终止核对。
        raise ValueError(f"doc_id mismatch: missing={sorted(missing)}, extra={sorted(extra)}")  # 列出缺失和多余的标识便于定位。
    for doc_id, row in source.items():  # 在标识相同的前提下逐条对比。
        changed = [key for key in COLUMNS if row[key] != target[doc_id][key]]  # 收集同一文档所有取值不同的列。
        if changed:  # 有字段差异时拒绝把迁移视为成功。
            raise ValueError(f"doc_id {doc_id}: differing fields: {', '.join(changed)}")  # 报告差异涉及的文档和列名。


def migrate(path: Path, settings: MySQLSettings) -> str:  # 从只读 SQLite 快照导入 MySQL 并核对结果。
    uri = path.resolve().as_uri() + "?mode=ro"  # 构造只读 URI，防止迁移修改源数据库。
    with closing(sqlite3.connect(uri, uri=True)) as source:  # 打开源文件并在读取后释放连接。
        source.row_factory = sqlite3.Row  # 让 SQLite 结果可按列名取值。
        rows = {row["doc_id"]: dict(row) for row in source.execute(SELECT_ROWS)}  # 按文档标识保存源表快照。

    DocumentRegistry(settings)  # DDL 在数据事务之前执行。
    with registry_transaction(settings) as cursor:  # 将目标数据的读取、写入和验证放在同一事务中。
        existing = target_rows(cursor)  # 读取写入前的目标快照。
        if existing:  # 目标已有数据时先核实是否与源完全相同。
            verify_rows(rows, existing)  # 已有记录必须与源逐字段一致。
            result = "already identical"  # 标记本次为重复执行且数据一致。
        else:  # 仅在目标为空时插入源记录。
            columns = ", ".join(COLUMNS)  # 按固定顺序生成 INSERT 列名。
            placeholders = ", ".join(["%s"] * len(COLUMNS))  # 为每列创建参数占位符。
            insert = f"INSERT INTO documents ({columns}) VALUES ({placeholders})"  # 组装参数化插入语句。
            for row in rows.values():  # 逐条复制源快照中的文档。
                cursor.execute(insert, tuple(row[key] for key in COLUMNS))  # 按列清单顺序绑定值并写入目标。
            verify_rows(rows, target_rows(cursor))  # 提交前重新读取目标并核对全部字段。
            result = "import verified"  # 记录本次导入及验证成功。
    counts = Counter(row["state"] for row in rows.values())  # 统计各状态的源记录数。
    return f"{result}: {len(rows)} documents, states={dict(counts)}"  # 返回迁移结果、文档总数和状态分布。


def main(argv=None) -> int:  # 运行迁移命令并返回进程退出码。
    parser = argparse.ArgumentParser(description=__doc__)  # 用模块说明生成命令帮助。
    parser.add_argument("sqlite_file", type=Path, help="停写后的 SQLite 文件或其备份")  # 要求调用方指定源 SQLite 文件。
    parser.add_argument("--config", type=Path, help="项目 YAML 配置文件")  # 允许指定目标库所在的配置文件。
    args = parser.parse_args(argv)  # 解析传入或命令行提供的参数。
    try:  # 统一处理读取配置与迁移失败的情况。
        settings = load_settings(config_path=args.config).mysql  # 从项目配置提取 MySQL 连接设置。
        print(migrate(args.sqlite_file, settings))  # 执行迁移并打印核对结果。
    except (ConfigurationError, sqlite3.Error, pymysql.MySQLError, RuntimeError, ValueError) as exc:  # 接收配置、数据库和数据一致性错误。
        print(f"registry migration failed: {exc}", file=sys.stderr)  # 将失败原因写入标准错误。
        return 1  # 用非零退出码标识迁移失败。
    return 0  # 迁移成功时返回零。


if __name__ == "__main__":  # 仅在直接运行脚本时启动迁移。
    raise SystemExit(main())  # 将入口返回值交给操作系统。
