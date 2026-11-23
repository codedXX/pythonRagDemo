"""Document registry backed by SQLite.

The registry tracks lifecycle state — which documents exist, whether they are
usable, and what went wrong when they are not. It deliberately does *not* store
chunk text: Milvus is the single source of truth for content, so that the
vector and keyword channels can never disagree about what exists.

States:

    staging -> active      ingestion finished and everything was written
    staging -> failed      ingestion aborted; nothing of it is retrievable
    active  -> (removed)   the row and all its content are gone

A document only becomes ``active`` after all of its units are written and
activated in Milvus, which is what makes ingestion atomic from the caller's
point of view.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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


class DocumentState(str, Enum):
    STAGING = "staging"
    ACTIVE = "active"
    FAILED = "failed"


@dataclass(slots=True)
class DocumentRecord:
    """One registered document."""

    doc_id: str
    name: str
    state: DocumentState
    parent_count: int = 0
    child_count: int = 0
    char_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.state is DocumentState.ACTIVE

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "name": self.name,
            "state": self.state.value,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "char_count": self.char_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


class DocumentRegistry:
    """Thread-safe SQLite registry.

    The ingestion pipeline runs on worker threads, so the connection is opened
    per operation with ``check_same_thread=False`` guarded by a lock rather than
    sharing one connection across threads.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    # ------------------------------------------------------------------ writes

    def begin(self, doc_id: str, name: str) -> DocumentRecord:
        """Register a document as being ingested."""
        with self._lock, self._connect() as connection:
            connection.execute(
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
                (doc_id, name, DocumentState.STAGING.value),
            )
        record = self.get(doc_id)
        assert record is not None
        return record

    def mark_active(
        self,
        doc_id: str,
        parent_count: int,
        child_count: int,
        char_count: int,
    ) -> None:
        """Record successful ingestion."""
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                   SET state = ?, parent_count = ?, child_count = ?,
                       char_count = ?, updated_at = datetime('now'), error = NULL
                 WHERE doc_id = ?
                """,
                (
                    DocumentState.ACTIVE.value,
                    parent_count,
                    child_count,
                    char_count,
                    doc_id,
                ),
            )

    def mark_failed(self, doc_id: str, error: str) -> None:
        """Record a failed ingestion together with the reason."""
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                   SET state = ?, error = ?, updated_at = datetime('now')
                 WHERE doc_id = ?
                """,
                (DocumentState.FAILED.value, error[:2000], doc_id),
            )

    def remove(self, doc_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # ------------------------------------------------------------------- reads

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return _to_record(row) if row else None

    def find_by_name(self, name: str) -> list[DocumentRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE name = ? ORDER BY created_at", (name,)
            ).fetchall()
        return [_to_record(row) for row in rows]

    def list_documents(self) -> list[DocumentRecord]:
        """Return every document, newest first.

        Failed rows are included on purpose: the interface shows them so a user
        learns why an upload did not work instead of seeing it vanish.
        """
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, name"
            ).fetchall()
        return [_to_record(row) for row in rows]

    def list_usable(self) -> list[DocumentRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE state = ? ORDER BY created_at DESC",
                (DocumentState.ACTIVE.value,),
            ).fetchall()
        return [_to_record(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS n FROM documents GROUP BY state"
            ).fetchall()
        return {row["state"]: row["n"] for row in rows}


def _to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        name=row["name"],
        state=DocumentState(row["state"]),
        parent_count=row["parent_count"],
        child_count=row["child_count"],
        char_count=row["char_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row["error"],
    )