"""Integration tests for the ingestion pipeline against real Milvus.

The embedding provider is faked so those tests need no API key, but Milvus is real:
the properties under test (atomicity, immediate visibility, replacement of same-name
documents) are properties of the store, and faking it would test nothing.
"""

from __future__ import annotations

import uuid

import pymupdf
import pytest

from rag_demo.chunking.chunker import Chunker
from rag_demo.config import load_settings
from rag_demo.ingestion import IngestionError, IngestionPipeline, Stage, remove_document
from rag_demo.store.milvus_store import MilvusStore
from rag_demo.store.registry import DocumentRegistry, DocumentState

pytestmark = pytest.mark.integration

DIM = 8


class FakeEmbedder:
    """Deterministic embeddings, so scores are reproducible and no key is needed."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def embed_texts(self, texts):
        self.calls += 1
        if self.fail_on and self.fail_on in " ".join(texts):
            raise RuntimeError("provider exploded")
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        # Cheap but content-sensitive: the same text always maps to the same point.
        seed = sum(ord(c) for c in text) % 97
        return [((seed + i) % 13) / 13.0 for i in range(DIM)]

    def probe_dimension(self, text: str = "ping") -> int:
        return DIM


def _milvus_available(uri: str) -> bool:
    try:
        from pymilvus import MilvusClient

        MilvusClient(uri=uri).get_server_version()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("ingest")
    settings = load_settings(root=root, use_env=False)
    if not _milvus_available(settings.milvus.uri):
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")
    suffix = uuid.uuid4().hex[:6]
    settings.milvus.child_collection = f"ing_child_{suffix}"
    settings.milvus.parent_collection = f"ing_parent_{suffix}"
    settings.milvus.meta_collection = f"ing_meta_{suffix}"
    settings.chunking.parent_target_chars = 300
    settings.chunking.parent_max_chars = 400
    settings.chunking.child_target_chars = 120
    settings.chunking.child_overlap_chars = 20
    settings.chunking.semantic_enabled = False
    return settings


@pytest.fixture(scope="module")
def store(settings):
    store = MilvusStore(settings)
    store.drop_all()
    store.ensure_collections(dimension=DIM)
    yield store
    store.drop_all()


@pytest.fixture()
def registry(tmp_path):
    return DocumentRegistry(tmp_path / "rag.db")


@pytest.fixture()
def pipeline(settings, store, registry):
    embedder = FakeEmbedder()
    return IngestionPipeline(
        settings,
        store,
        registry,
        embedder,
        Chunker(settings.chunking, embedder=embedder),
    )


def _markdown(body: str) -> bytes:
    return body.encode("utf-8")


def _pdf(pages: list[str]) -> bytes:
    document = pymupdf.open()
    for body in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), body, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def _clean(store, registry):
    store.client.delete(store.child_name, filter='doc_id != ""')
    store.client.delete(store.parent_name, filter='doc_id != ""')
    for record in registry.list_documents():
        registry.remove(record.doc_id)


class TestHappyPath:
    def test_markdown_ingestion_produces_retrievable_units(self, pipeline, store, registry):
        body = "# Section\n\n" + "\n\n".join(f"Paragraph {i} with meaningful content." for i in range(6))

        result = pipeline.ingest("notes.md", _markdown(body))

        assert result.parent_count >= 1
        assert result.child_count >= result.parent_count
        record = registry.get(result.doc_id)
        assert record is not None
        assert record.state is DocumentState.ACTIVE
        assert record.child_count == result.child_count
        _clean(store, registry)

    def test_pdf_ingestion_works(self, pipeline, store, registry):
        body = "A substantial paragraph of text for ingestion. " * 12

        result = pipeline.ingest("paper.pdf", _pdf([body, body]))

        assert result.child_count > 0
        _clean(store, registry)

    def test_units_are_retrievable_immediately(self, pipeline, store, registry):
        """The Strong-consistency promise, at the pipeline level."""
        vector = FakeEmbedder._vector("hello world")
        body = "hello world " * 40

        pipeline.ingest("immediate.md", _markdown(body))

        hits = store.search_children(vector, limit=5)
        assert hits
        _clean(store, registry)

    def test_progress_is_reported_in_order(self, pipeline, store, registry):
        seen: list[str] = []

        pipeline.ingest(
            "progress.md",
            _markdown("content " * 100),
            on_progress=lambda p: seen.append(p.stage.value),
        )

        assert seen[0] == Stage.PARSING.value
        assert Stage.CHUNKING.value in seen
        assert Stage.EMBEDDING.value in seen
        assert Stage.PERSISTING.value in seen
        assert seen[-1] == Stage.DONE.value
        _clean(store, registry)

    def test_char_count_is_recorded(self, pipeline, store, registry):
        result = pipeline.ingest("count.md", _markdown("word " * 200))

        assert result.char_count > 0
        assert registry.get(result.doc_id).char_count == result.char_count
        _clean(store, registry)


class TestAtomicity:
    def test_failed_ingestion_leaves_nothing_retrievable(self, settings, store, registry):
        """A failure at embedding must not leave a half-indexed document."""
        embedder = FakeEmbedder(fail_on="EXPLODE")
        pipeline = IngestionPipeline(
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)
        )

        with pytest.raises(IngestionError):
            pipeline.ingest("broken.md", _markdown("EXPLODE " * 100))

        assert store.search_children(FakeEmbedder._vector("EXPLODE"), limit=10) == []
        assert store.all_active_children() == []

    def test_failure_is_recorded_with_its_stage(self, settings, store, registry):
        embedder = FakeEmbedder(fail_on="EXPLODE")
        pipeline = IngestionPipeline(
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)
        )

        with pytest.raises(IngestionError) as excinfo:
            pipeline.ingest("broken.md", _markdown("EXPLODE " * 100))

        assert excinfo.value.stage is Stage.EMBEDDING
        records = registry.list_documents()
        assert records[0].state is DocumentState.FAILED
        assert Stage.EMBEDDING.value in (records[0].error or "")

    def test_scanned_pdf_fails_before_any_embedding_call(
        self, settings, store, registry
    ):
        """Parsing failures must not spend money on embedding."""
        embedder = FakeEmbedder()
        pipeline = IngestionPipeline(
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)
        )
        blank = pymupdf.open()
        blank.new_page()
        content = blank.tobytes()
        blank.close()

        with pytest.raises(IngestionError) as excinfo:
            pipeline.ingest("scan.pdf", content)

        assert excinfo.value.stage is Stage.PARSING
        assert embedder.calls == 0

    def test_unsupported_type_fails_without_embedding(self, settings, store, registry):
        embedder = FakeEmbedder()
        pipeline = IngestionPipeline(
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)
        )

        with pytest.raises(IngestionError) as excinfo:
            pipeline.ingest("notes.txt", b"plain text")

        assert excinfo.value.stage is Stage.PARSING
        assert embedder.calls == 0


class TestReplacement:
    def test_reuploading_same_name_replaces_rather_than_duplicates(
        self, pipeline, store, registry
    ):
        pipeline.ingest("same.md", _markdown("original content " * 40))
        pipeline.ingest("same.md", _markdown("replacement content " * 40))

        active = store.all_active_children()

        assert active
        assert all("replacement" in row["text"] for row in active)
        assert len([r for r in registry.list_usable() if r.name == "same.md"]) == 1
        _clean(store, registry)

    def test_changed_content_gets_a_new_document_id(self, pipeline, store, registry):
        first = pipeline.ingest("doc.md", _markdown("version one " * 40))
        second = pipeline.ingest("doc.md", _markdown("version two " * 40))

        # The document id follows the bytes, so a changed file is a new identity
        # while the old one is removed by name.
        assert first.doc_id != second.doc_id
        assert len(registry.list_usable()) == 1
        _clean(store, registry)


class TestRemoval:
    def test_remove_document_deletes_content_and_record(self, pipeline, store, registry):
        result = pipeline.ingest("gone.md", _markdown("content to remove " * 40))

        assert remove_document(result.doc_id, store, registry) is True

        assert store.search_children(FakeEmbedder._vector("content"), limit=10) == []
        assert registry.get(result.doc_id) is None

    def test_remove_unknown_document_reports_false(self, store, registry):
        assert remove_document("does-not-exist", store, registry) is False