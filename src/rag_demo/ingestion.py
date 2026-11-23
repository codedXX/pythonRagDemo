"""The ingestion pipeline: bytes in, retrievable units out.

The ordering here is deliberate:

1. Parse, so a scanned PDF or unsupported file fails before any API call is spent.
2. Chunk, so the number of embedding calls is known before starting.
3. Embed, which is the slow and costly step.
4. Persist inactively, then flip active, so a failure anywhere leaves nothing
   retrievable.

Stages are reported through a callback because the interface shows progress; a
30-page PDF takes tens of seconds and an unresponsive progress bar is
indistinguishable from a hang.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from rag_demo.chunking.chunker import ChunkSet, Chunker
from rag_demo.config import Settings
from rag_demo.embedding.provider import Embedder
from rag_demo.parsers import ParsedDocument, parse_document
from rag_demo.parsers.base import document_id
from rag_demo.store.milvus_store import ChildUnit, MilvusStore, ParentUnit, normalise
from rag_demo.store.registry import DocumentRegistry


class Stage(str, Enum):
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    PERSISTING = "persisting"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class Progress:
    """One progress notification."""

    stage: Stage
    message: str
    current: int = 0
    total: int = 0

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.current / self.total)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "fraction": round(self.fraction, 4),
        }


ProgressCallback = Callable[[Progress], None]


@dataclass(slots=True)
class IngestResult:
    """What an ingestion produced."""

    doc_id: str
    name: str
    parent_count: int
    child_count: int
    char_count: int
    duration_seconds: float


class IngestionError(RuntimeError):
    """Ingestion failed, with the stage that failed attached."""

    def __init__(self, stage: Stage, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"{stage.value}: {message}")


class IngestionPipeline:
    """Turns uploaded bytes into retrievable parent/child units."""

    def __init__(
        self,
        settings: Settings,
        store: MilvusStore,
        registry: DocumentRegistry,
        embedder: Embedder,
        chunker: Chunker | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._chunker = chunker or Chunker(settings.chunking, embedder=embedder)

    def ingest(
        self,
        name: str,
        content: bytes,
        on_progress: ProgressCallback | None = None,
        doc_id: str | None = None,
    ) -> IngestResult:
        """Ingest one document.

        Raises :class:`IngestionError` with the failing stage when something goes
        wrong; the registry records the same information for later inspection.
        """
        started = time.time()
        notify = on_progress or (lambda _progress: None)
        identifier = doc_id or document_id(name, content)

        self._registry.begin(identifier, name)

        try:
            parsed = self._parse(name, content, notify)
            chunks = self._chunk(parsed, identifier, notify)
            children, parents = self._embed_and_persist(identifier, chunks, notify)
        except IngestionError as error:
            self._registry.mark_failed(identifier, str(error))
            notify(Progress(stage=Stage.FAILED, message=str(error)))
            raise
        except Exception as error:  # unexpected: still record it
            message = f"{type(error).__name__}: {error}"
            self._registry.mark_failed(identifier, message)
            notify(Progress(stage=Stage.FAILED, message=message))
            raise IngestionError(Stage.PERSISTING, message) from error

        duration = time.time() - started
        notify(
            Progress(
                stage=Stage.DONE,
                message=f"indexed {len(parents)} parents and {len(children)} children",
                current=1,
                total=1,
            )
        )
        return IngestResult(
            doc_id=identifier,
            name=name,
            parent_count=len(parents),
            child_count=len(children),
            char_count=chunks.char_count,
            duration_seconds=duration,
        )

    # -------------------------------------------------------------------- stages

    def _parse(
        self, name: str, content: bytes, notify: ProgressCallback
    ) -> ParsedDocument:
        notify(Progress(stage=Stage.PARSING, message=f"parsing {name}"))
        try:
            document = parse_document(name, content, self._settings.parsing)
        except Exception as error:
            raise IngestionError(Stage.PARSING, str(error)) from error

        if document.is_empty():
            raise IngestionError(
                Stage.PARSING,
                f"{name} produced no extractable text",
            )
        return document

    def _chunk(
        self, document: ParsedDocument, identifier: str, notify: ProgressCallback
    ) -> ChunkSet:
        notify(
            Progress(
                stage=Stage.CHUNKING,
                message=f"chunking {document.name}",
            )
        )
        try:
            chunks = self._chunker.chunk(document, identifier)
        except Exception as error:
            raise IngestionError(Stage.CHUNKING, str(error)) from error

        if not chunks.children:
            raise IngestionError(
                Stage.CHUNKING, f"{document.name} produced no retrievable units"
            )
        return chunks

    def _embed_and_persist(
        self, identifier: str, chunks: ChunkSet, notify: ProgressCallback
    ) -> tuple[list[ChildUnit], list[ParentUnit]]:
        texts = [child.text for child in chunks.children]
        notify(
            Progress(
                stage=Stage.EMBEDDING,
                message=f"embedding {len(texts)} units",
                current=0,
                total=len(texts),
            )
        )

        try:
            vectors = self._embedder.embed_texts(texts)
        except Exception as error:
            raise IngestionError(Stage.EMBEDDING, str(error)) from error

        if len(vectors) != len(texts):
            raise IngestionError(
                Stage.EMBEDDING,
                f"asked for {len(texts)} vectors but received {len(vectors)}",
            )

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise IngestionError(
                Stage.EMBEDDING, f"inconsistent vector dimensions: {sorted(dimensions)}"
            )

        notify(
            Progress(
                stage=Stage.EMBEDDING,
                message=f"embedded {len(vectors)} units",
                current=len(vectors),
                total=len(vectors),
            )
        )

        children = [
            ChildUnit(
                id=child.id,
                doc_id=identifier,
                parent_id=child.parent_id,
                child_idx=child.child_idx,
                text=child.text,
                breadcrumb=child.breadcrumb,
                page=child.page,
                source_name=child.source_name,
                # Normalised so stored vectors are comparable by dot product as
                # well as by the collection's cosine metric.
                embedding=normalise(vector),
            )
            for child, vector in zip(chunks.children, vectors, strict=True)
        ]
        parents = [
            ParentUnit(
                parent_id=parent.parent_id,
                doc_id=identifier,
                text=parent.text,
                breadcrumb=parent.breadcrumb,
                page_start=parent.page_start,
                source_name=parent.source_name,
            )
            for parent in chunks.parents
        ]

        notify(
            Progress(
                stage=Stage.PERSISTING,
                message=f"writing {len(parents)} parents and {len(children)} children",
            )
        )
        try:
            # Same-name documents are replaced rather than duplicated. Deleting
            # first keeps a failed re-ingest from leaving the old version plus a
            # partial new one.
            for existing in self._registry.find_by_name(chunks.source_name):
                if existing.doc_id != identifier:
                    self._store.delete_document(existing.doc_id)
                    self._registry.remove(existing.doc_id)

            self._store.delete_document(identifier)
            self._store.insert_units(children, parents)
            self._store.activate_document(identifier)
        except Exception as error:
            raise IngestionError(Stage.PERSISTING, str(error)) from error

        self._registry.mark_active(
            identifier,
            parent_count=len(parents),
            child_count=len(children),
            char_count=chunks.char_count,
        )
        return children, parents


def remove_document(
    doc_id: str, store: MilvusStore, registry: DocumentRegistry
) -> bool:
    """Remove a document and everything of it that is retrievable."""
    record = registry.get(doc_id)
    if record is None:
        return False
    store.delete_document(doc_id)
    registry.remove(doc_id)
    return True