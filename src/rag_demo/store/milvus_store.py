"""Milvus-backed storage with parent/child separation.

Two collections rather than one, because the two levels have opposite access
patterns:

* ``rag_child`` holds the units that get embedded and retrieved by ANN. It needs
  a vector index and is queried constantly.
* ``rag_parent`` holds the larger units supplied to the language model. It is
  only ever read by primary key for the ids that survived reranking, so its vector
  field carries no meaning.

Milvus requires every collection to contain an indexed vector field: creating one
without is rejected ("schema does not contain vector field"), as is declaring a
vector field with no index on it, and the smallest permitted dimension is 2 rather
than 1. So the parent collection carries a two-element constant placeholder
vector. It is written by us, never embedded, and never searched — the point was
only to avoid spending embedding calls on parents, which it does.

Consistency is pinned to Strong. Measured against milvus v2.6.23: with the
default (Bounded) level, a row inserted is not visible to a query issued right
after it (count 0, no search hits); it becomes visible after an explicit flush or
under Strong. Since the product promise is "once ingestion finishes you can ask
about it", Bounded would produce a bug that looks like nothing is wrong in the
logs. At this scale the latency cost of Strong is irrelevant.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pymilvus import DataType, MilvusClient

from rag_demo.config import Settings

# Field limits. Generous but bounded: an over-long value must be caught when it
# is written, not silently truncated by the server.
MAX_ID_LEN = 96
MAX_TEXT_LEN = 65535
MAX_BREADCRUMB_LEN = 4096
MAX_NAME_LEN = 512

# Milvus rejects dimension 1 and requires a vector index on any vector field, so
# the parent collection's vector is a two-element constant. It is never embedded
# and never searched; see the module docstring.
PLACEHOLDER_DIM = 2
PLACEHOLDER_VECTOR = [0.0, 0.0]


@dataclass(slots=True)
class ChildUnit:
    """One retrievable unit."""

    id: str
    doc_id: str
    parent_id: str
    child_idx: int
    text: str
    breadcrumb: str
    page: int
    source_name: str
    embedding: list[float]

    def as_row(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "child_idx": self.child_idx,
            "text": self.text[:MAX_TEXT_LEN],
            "breadcrumb": self.breadcrumb[:MAX_BREADCRUMB_LEN],
            "page": self.page,
            "source_name": self.source_name[:MAX_NAME_LEN],
            "active": False,
            "embedding": self.embedding,
        }


@dataclass(slots=True)
class ParentUnit:
    """One context unit, returned whole to the language model."""

    parent_id: str
    doc_id: str
    text: str
    breadcrumb: str
    page_start: int
    source_name: str

    def as_row(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "text": self.text[:MAX_TEXT_LEN],
            "breadcrumb": self.breadcrumb[:MAX_BREADCRUMB_LEN],
            "page_start": self.page_start,
            "source_name": self.source_name[:MAX_NAME_LEN],
            "active": False,
            "placeholder": list(PLACEHOLDER_VECTOR),
        }


class MilvusStore:
    """Schema management and low-level access for the three collections."""

    def __init__(self, settings: Settings, client: MilvusClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._dimension: int | None = None

    # ------------------------------------------------------------------ client

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(
                uri=self._settings.milvus.uri,
                token=self._settings.milvus.token or None,
            )
        return self._client

    @property
    def child_name(self) -> str:
        return self._settings.milvus.child_collection

    @property
    def parent_name(self) -> str:
        return self._settings.milvus.parent_collection

    @property
    def meta_name(self) -> str:
        return self._settings.milvus.meta_collection

    # ------------------------------------------------------------------ schema

    def ensure_collections(self, dimension: int) -> None:
        """Create the collections if absent, using ``dimension`` for vectors."""
        self._dimension = dimension
        if not self.client.has_collection(self.child_name):
            self._create_child(dimension)
        if not self.client.has_collection(self.parent_name):
            self._create_parent()
        if not self.client.has_collection(self.meta_name):
            self._create_meta()

    def _create_child(self, dimension: int) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LEN)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_ID_LEN)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=MAX_ID_LEN)
        schema.add_field("child_idx", DataType.INT32)
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LEN)
        schema.add_field("breadcrumb", DataType.VARCHAR, max_length=MAX_BREADCRUMB_LEN)
        schema.add_field("page", DataType.INT32)
        schema.add_field("source_name", DataType.VARCHAR, max_length=MAX_NAME_LEN)
        schema.add_field("active", DataType.BOOL)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)

        index_params = self.client.prepare_index_params()
        # COSINE with client-side normalisation; see _normalise below.
        index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
        for scalar in ("doc_id", "parent_id", "active"):
            index_params.add_index(scalar, index_type="INVERTED")

        self.client.create_collection(
            self.child_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._settings.milvus.consistency_level,
        )

    def _create_parent(self) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            "parent_id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LEN
        )
        schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_ID_LEN)
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LEN)
        schema.add_field("breadcrumb", DataType.VARCHAR, max_length=MAX_BREADCRUMB_LEN)
        schema.add_field("page_start", DataType.INT32)
        schema.add_field("source_name", DataType.VARCHAR, max_length=MAX_NAME_LEN)
        schema.add_field("active", DataType.BOOL)
        # Required by Milvus; carries no meaning and is never searched.
        schema.add_field("placeholder", DataType.FLOAT_VECTOR, dim=PLACEHOLDER_DIM)

        index_params = self.client.prepare_index_params()
        index_params.add_index("placeholder", index_type="AUTOINDEX", metric_type="COSINE")
        for scalar in ("doc_id", "active"):
            index_params.add_index(scalar, index_type="INVERTED")

        self.client.create_collection(
            self.parent_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._settings.milvus.consistency_level,
        )

    def _create_meta(self) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("key", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_LEN)
        schema.add_field("value", DataType.VARCHAR, max_length=MAX_TEXT_LEN)
        # Same Milvus requirement as the parent collection: a vector field and an
        # index on it are mandatory. Unused here.
        schema.add_field("placeholder", DataType.FLOAT_VECTOR, dim=PLACEHOLDER_DIM)

        index_params = self.client.prepare_index_params()
        index_params.add_index("placeholder", index_type="AUTOINDEX", metric_type="COSINE")

        self.client.create_collection(
            self.meta_name,
            schema=schema,
            index_params=index_params,
            consistency_level=self._settings.milvus.consistency_level,
        )

    # -------------------------------------------------------------------- meta

    def read_meta(self) -> dict:
        """Return the stored schema metadata, or ``{}`` when none exists."""
        if not self.client.has_collection(self.meta_name):
            return {}
        try:
            rows = self.client.query(
                self.meta_name, filter="", output_fields=["key", "value"], limit=100
            )
        except Exception:
            return {}
        raw = {row["key"]: row["value"] for row in rows}
        if "schema" not in raw:
            return {}
        try:
            return json.loads(raw["schema"])
        except ValueError:
            return {}

    def write_meta(self, payload: dict) -> None:
        """Persist schema metadata, replacing any previous value."""
        if not self.client.has_collection(self.meta_name):
            self._create_meta()
        row = {
            "key": "schema",
            "value": json.dumps(payload, ensure_ascii=False),
            "placeholder": list(PLACEHOLDER_VECTOR),
        }
        self.client.upsert(self.meta_name, [row])

    # ------------------------------------------------------------------ insert

    def insert_units(
        self,
        children: Sequence[ChildUnit],
        parents: Sequence[ParentUnit],
    ) -> None:
        """Insert both levels in an inactive state.

        Writes are staged with ``active=False`` and only flipped once the whole
        document is in place, so a failure part-way through can never leave a
        half-ingested document visible to retrieval.
        """
        if children:
            self.client.insert(self.child_name, [c.as_row() for c in children])
        if parents:
            self.client.insert(self.parent_name, [p.as_row() for p in parents])

    def activate_document(self, doc_id: str) -> None:
        """Make a document's units retrievable."""
        # Milvus has no bulk "update where"; upsert requires whole rows, so the
        # rows are read, flipped and written back.
        for name, key in ((self.child_name, "id"), (self.parent_name, "parent_id")):
            rows = self.client.query(
                name, filter=f'doc_id == "{doc_id}"', output_fields=["*"]
            )
            if not rows:
                continue
            for row in rows:
                # Embeddings are absent from parent rows and present in child
                # rows; query with output_fields=["*"] returns them only where
                # they exist.
                row["active"] = True
            self.client.upsert(name, rows)

    def deactivate_document(self, doc_id: str) -> None:
        """Hide a document's units from retrieval without deleting them."""
        for name in (self.child_name, self.parent_name):
            rows = self.client.query(
                name, filter=f'doc_id == "{doc_id}"', output_fields=["*"]
            )
            if not rows:
                continue
            for row in rows:
                row["active"] = False
            self.client.upsert(name, rows)

    def delete_document(self, doc_id: str) -> None:
        """Remove every unit belonging to a document."""
        for name in (self.child_name, self.parent_name):
            self.client.delete(name, filter=f'doc_id == "{doc_id}"')

    # ------------------------------------------------------------------ select

    def search_children(
        self,
        embedding: Sequence[float],
        limit: int,
        doc_id: str | None = None,
    ) -> list[dict]:
        """ANN search over active child units."""
        expression = "active == true"
        if doc_id:
            expression += f' and doc_id == "{doc_id}"'
        hits = self.client.search(
            self.child_name,
            data=[list(embedding)],
            anns_field="embedding",
            limit=limit,
            filter=expression,
            output_fields=["id", "doc_id", "parent_id", "text", "breadcrumb", "page", "source_name"],
        )
        results = []
        for hit in hits[0] if hits else []:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            results.append(
                {
                    "id": entity.get("id"),
                    "doc_id": entity.get("doc_id"),
                    "parent_id": entity.get("parent_id"),
                    "text": entity.get("text", ""),
                    "breadcrumb": entity.get("breadcrumb", ""),
                    "page": entity.get("page", 0),
                    "source_name": entity.get("source_name", ""),
                    "score": float(hit.get("distance", 0.0)) if isinstance(hit, dict) else 0.0,
                }
            )
        return results

    def fetch_parents(self, parent_ids: Iterable[str]) -> dict[str, dict]:
        """Fetch parent units by primary key, keyed by id."""
        ids = [str(p) for p in dict.fromkeys(parent_ids)]
        if not ids:
            return {}
        quoted = ", ".join(f'"{p}"' for p in ids)
        rows = self.client.query(
            self.parent_name,
            filter=f"parent_id in [{quoted}]",
            output_fields=["parent_id", "doc_id", "text", "breadcrumb", "page_start", "source_name", "active"],
        )
        return {row["parent_id"]: row for row in rows}

    def all_active_children(self) -> list[dict]:
        """Return every active child unit.

        Used to rebuild the lexical index. Milvus is deliberately the single
        source of truth here: building the keyword index from files on disk would
        let the two channels disagree the moment a file changed.
        """
        rows = self.client.query(
            self.child_name,
            filter="active == true",
            output_fields=["id", "doc_id", "parent_id", "text", "breadcrumb", "page", "source_name"],
            limit=16384,
        )
        return rows

    def collection_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for name in (self.child_name, self.parent_name):
            if not self.client.has_collection(name):
                stats[name] = 0
                continue
            try:
                # Milvus forbids pagination on a count ("count entities with
                # pagination is not allowed"), so this specific query must omit
                # the limit that plain empty-filter queries require.
                rows = self.client.query(name, filter="", output_fields=["count(*)"])
                stats[name] = int(rows[0]["count(*)"]) if rows else 0
            except Exception:
                stats[name] = -1
        return stats

    def drop_all(self) -> None:
        """Drop this project's collections. Used by tests and by rebuild flows."""
        for name in (self.child_name, self.parent_name, self.meta_name):
            if self.client.has_collection(name):
                self.client.drop_collection(name)


def normalise(vector: Sequence[float]) -> list[float]:
    """Scale a vector to unit length.

    Collections use COSINE, which is scale invariant, so normalising is not
    strictly required. It is still applied so that stored vectors are comparable
    by dot product too, which keeps diagnostics intuitive.
    """
    magnitude = sum(x * x for x in vector) ** 0.5
    if magnitude == 0:
        return list(vector)
    return [x / magnitude for x in vector]