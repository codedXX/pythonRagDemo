"""停写后将 SQLite 文档登记表导入配置的本地 MySQL。"""

import argparse
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

import pymysql

from rag_demo.config import ConfigurationError, MySQLSettings, load_settings
from rag_demo.store.registry import DocumentRegistry, registry_transaction

COLUMNS = ("doc_id", "name", "state", "parent_count", "child_count", "char_count",
           "created_at", "updated_at", "error")
SELECT_ROWS = "SELECT " + ", ".join(COLUMNS) + " FROM documents"


def target_rows(cursor) -> dict:
    cursor.execute(SELECT_ROWS)
    rows = {}
    for row in cursor.fetchall():
        for key in ("created_at", "updated_at"):
            row[key] = row[key].isoformat(sep=" ")
        rows[row["doc_id"]] = row
    return rows


def verify_rows(source: dict, target: dict) -> None:
    missing = source.keys() - target.keys()
    extra = target.keys() - source.keys()
    if missing or extra:
        raise ValueError(f"doc_id mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for doc_id, row in source.items():
        changed = [key for key in COLUMNS if row[key] != target[doc_id][key]]
        if changed:
            raise ValueError(f"doc_id {doc_id}: differing fields: {', '.join(changed)}")


def migrate(path: Path, settings: MySQLSettings) -> str:
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        source.row_factory = sqlite3.Row
        rows = {row["doc_id"]: dict(row) for row in source.execute(SELECT_ROWS)}

    DocumentRegistry(settings)  # DDL 在数据事务之前执行。
    with registry_transaction(settings) as cursor:
        existing = target_rows(cursor)
        if existing:
            verify_rows(rows, existing)
            result = "already identical"
        else:
            columns = ", ".join(COLUMNS)
            placeholders = ", ".join(["%s"] * len(COLUMNS))
            insert = f"INSERT INTO documents ({columns}) VALUES ({placeholders})"
            for row in rows.values():
                cursor.execute(insert, tuple(row[key] for key in COLUMNS))
            verify_rows(rows, target_rows(cursor))
            result = "import verified"
    counts = Counter(row["state"] for row in rows.values())
    return f"{result}: {len(rows)} documents, states={dict(counts)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_file", type=Path, help="停写后的 SQLite 文件或其备份")
    parser.add_argument("--config", type=Path, help="项目 YAML 配置文件")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(config_path=args.config).mysql
        print(migrate(args.sqlite_file, settings))
    except (ConfigurationError, sqlite3.Error, pymysql.MySQLError, RuntimeError, ValueError) as exc:
        print(f"registry migration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
