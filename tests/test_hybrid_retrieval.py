"""Tests for the full hybrid retrieval pipeline against real Milvus.

Embeddings and reranking are faked so the suite needs no API key, but the store is
real: deduplication, parent expansion and the immediate-visibility property are
store behaviours that a fake would not exercise.
"""

from __future__ import annotations

import uuid

import pytest

from rag_demo.config import load_settings
from rag_demo.embedding.rerank import RerankResult
from rag_demo.errors import TransientFailure
from rag_demo.retrieval.hybrid import Context, HybridRetriever
from rag_demo.retrieval.lexical import LexicalIndex
from rag_demo.store.milvus_store import ChildUnit, MilvusStore, ParentUnit, normalise

pytestmark = pytest.mark.integration

DIM = 6


def _vector(seed: int) -> list[float]:
    return normalise([((seed + i * 7) % 17) + 1.0 for i in range(DIM)])


class FakeEmbedder:
    """Deterministic embeddings keyed by content, so scores reproduce exactly."""

    def embed_texts(self, texts):
        return [_vector(sum(ord(c) for c in text) % 50) for text in texts]


class FakeReranker:
    """Reranks by a keyword rule so the ordering is predictable and inspectable."""

    def __init__(self, keyword: str = "IMPORTANT", failing: bool = False) -> None:
        self.keyword = keyword
        self.failing = failing
        self.documents: list[str] = []

    def rerank(self, query, documents, top_n=None):
        self.documents = list(documents)
        if self.failing:
            raise TransientFailure("rerank", "service unavailable")
        scored = [
            RerankResult(index=i, score=1.0 if self.keyword in document else 0.1)
            for i, document in enumerate(documents)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored


def _milvus_available(uri: str) -> bool:
    try:
        from pymilvus import MilvusClient

        MilvusClient(uri=uri).get_server_version()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrieval")
    settings = load_settings(root=root, use_env=False)
    if not _milvus_available(settings.milvus.uri):
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")
    suffix = uuid.uuid4().hex[:6]
    settings.milvus.child_collection = f"ret_child_{suffix}"
    settings.milvus.parent_collection = f"ret_parent_{suffix}"
    settings.milvus.meta_collection = f"ret_meta_{suffix}"
    settings.retrieval.dense_top_k = 20
    settings.retrieval.sparse_top_k = 20
    settings.retrieval.rerank_candidates = 10
    settings.retrieval.context_count = 4
    settings.retrieval.context_min_count = 2
    settings.retrieval.context_char_budget = 4000
    settings.retrieval.parent_excerpt_chars = 500
    return settings


@pytest.fixture(scope="module")
def store(settings):
    store = MilvusStore(settings)
    store.drop_all()
    store.ensure_collections(dimension=DIM)
    yield store
    store.drop_all()


@pytest.fixture(autouse=True)
def clean(store):
    store.client.delete(store.child_name, filter='doc_id != ""')
    store.client.delete(store.parent_name, filter='doc_id != ""')
    yield


def _populate(store, doc_id: str, parents: dict[str, tuple[str, str]], children_per_parent: int = 2):
    """Insert one document with the given parents and generated children."""
    child_units = []
    parent_units = []
    for index, (parent_id, (parent_text, breadcrumb)) in enumerate(parents.items()):
        parent_units.append(
            ParentUnit(
                parent_id=parent_id,
                doc_id=doc_id,
                text=parent_text,
                breadcrumb=breadcrumb,
                page_start=index + 1,
                source_name=f"{doc_id}.pdf",
            )
        )
        for child_idx in range(children_per_parent):
            text = f"{parent_text} part {child_idx}"
            child_units.append(
                ChildUnit(
                    id=f"{parent_id}_c{child_idx}",
                    doc_id=doc_id,
                    parent_id=parent_id,
                    child_idx=child_idx,
                    text=text,
                    breadcrumb=breadcrumb,
                    page=index + 1,
                    source_name=f"{doc_id}.pdf",
                    embedding=_vector(sum(ord(c) for c in text) % 50),
                )
            )
    store.insert_units(child_units, parent_units)
    store.activate_document(doc_id)


def _retriever(settings, store, reranker=None):
    lexical = LexicalIndex()
    retriever = HybridRetriever(
        settings.retrieval, store, FakeEmbedder(), lexical, reranker
    )
    retriever.rebuild_lexical_index()
    return retriever


class TestRecall:
    def test_sparse_channel_finds_keyword_match(self, settings, store):
        _populate(
            store,
            "d1",
            {
                "d1_p0": ("混合检索需要稀疏召回", "ch1"),
                "d1_p1": ("今天是晴朗的好天气", "ch2"),
            },
        )
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("混合检索")

        assert outcome.sparse
        assert any("混合检索" in c.text for c in outcome.sparse)

    def test_dense_channel_returns_results(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("some content about retrieval", "ch1")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("retrieval")

        assert outcome.dense

    def test_only_active_units_are_recalled(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("active content here", "ch1")})
        # Insert a second document but never activate it.
        store.insert_units(
            [
                ChildUnit(
                    id="d2_p0_c0",
                    doc_id="d2",
                    parent_id="d2_p0",
                    child_idx=0,
                    text="inactive content here",
                    breadcrumb="ch1",
                    page=1,
                    source_name="d2.pdf",
                    embedding=_vector(9),
                )
            ],
            [
                ParentUnit(
                    parent_id="d2_p0",
                    doc_id="d2",
                    text="inactive content here",
                    breadcrumb="ch1",
                    page_start=1,
                    source_name="d2.pdf",
                )
            ],
        )
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("inactive content")

        assert all(c.doc_id != "d2" for c in outcome.fused)

    def test_one_channel_empty_still_returns_results(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("only lexical will match zzz", "ch1")})
        retriever = _retriever(settings, store)

        # A query of only stopwords gives the lexical channel nothing to work with.
        outcome = retriever.retrieve("的 了 是")

        assert outcome.sparse == []
        assert outcome.contexts or outcome.fused

    def test_empty_knowledge_base_returns_nothing(self, settings, store):
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("anything")

        assert outcome.empty
        assert outcome.contexts == []


class TestFusion:
    def test_candidate_found_by_both_channels_ranks_first(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("混合检索 hybrid search explained", "ch1")})
        _populate(store, "d2", {"d2_p0": ("unrelated material entirely", "ch1")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("混合检索")

        top = outcome.fused[0]
        assert "dense" in top.channels

    def test_fused_positions_are_contiguous(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("alpha beta gamma", "ch1"), "d1_p1": ("delta epsilon", "ch2")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("alpha")

        assert [c.fused_rank for c in outcome.fused] == list(range(len(outcome.fused)))


class TestRerank:
    def test_reranker_reorders_candidates(self, settings, store):
        _populate(
            store,
            "d1",
            {
                "d1_p0": ("ordinary content about search", "ch1"),
                "d1_p1": ("IMPORTANT content about search", "ch2"),
            },
        )
        retriever = _retriever(settings, store, FakeReranker(keyword="IMPORTANT"))

        outcome = retriever.retrieve("search")

        assert outcome.rerank_applied is True
        assert "IMPORTANT" in outcome.contexts[0].text

    def test_rerank_input_includes_breadcrumb(self, settings, store):
        """The reranker must see provenance; bare fragments score unfairly."""
        _populate(store, "d1", {"d1_p0": ("body text", "Chapter Three")})
        reranker = FakeReranker()
        retriever = _retriever(settings, store, reranker)

        retriever.retrieve("body")

        assert reranker.documents
        assert all("Chapter Three" in document for document in reranker.documents)

    def test_more_candidates_are_reranked_than_supplied(self, settings, store):
        """Dedup shrinks the supply, so the rerank pool must be larger."""
        for index in range(8):
            _populate(
                store,
                f"d{index}",
                {f"d{index}_p0": (f"content number {index} about topic", f"ch{index}")},
                children_per_parent=1,
            )
        reranker = FakeReranker()
        retriever = _retriever(settings, store, reranker)

        outcome = retriever.retrieve("content topic")

        assert len(reranker.documents) > len(outcome.contexts)

    def test_reranker_failure_falls_back_to_fused_order(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("content that should still be returned", "ch1")})
        retriever = _retriever(settings, store, FakeReranker(failing=True))

        outcome = retriever.retrieve("content")

        assert outcome.rerank_applied is False
        assert outcome.rerank_error
        # The answer must still be answerable rather than empty.
        assert outcome.contexts

    def test_missing_reranker_is_not_an_error(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("content here", "ch1")})
        retriever = _retriever(settings, store, reranker=None)

        outcome = retriever.retrieve("content")

        assert outcome.rerank_applied is False
        assert outcome.contexts


class TestDeduplication:
    def test_siblings_collapse_to_one_context(self, settings, store):
        # One parent with four children, all matching the query.
        _populate(store, "d1", {"d1_p0": ("shared topic content", "ch1")}, children_per_parent=4)
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("shared topic content")

        parent_ids = [c.parent_id for c in outcome.contexts]
        assert len(parent_ids) == len(set(parent_ids)) == 1

    def test_supply_is_backfilled_after_collapse(self, settings, store):
        """Duplicate suppression must not silently reduce the context count."""
        # Three parents, each with three matching children: a naive top-k would
        # return three contexts all from the same parent.
        for index in range(3):
            _populate(
                store,
                f"d{index}",
                {f"d{index}_p0": (f"topic content family {index}", "ch1")},
                children_per_parent=3,
            )
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("topic content family")

        assert len(outcome.contexts) >= 2
        assert len({c.parent_id for c in outcome.contexts}) == len(outcome.contexts)

    def test_context_count_is_capped(self, settings, store):
        for index in range(10):
            _populate(store, f"d{index}", {f"d{index}_p0": (f"material {index} shared word", "ch")}, children_per_parent=1)
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("shared word")

        assert len(outcome.contexts) <= settings.retrieval.context_count


class TestExpansion:
    def test_context_uses_parent_text_not_child_text(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("the whole parent passage", "ch1")})

        retriever = _retriever(settings, store)
        outcome = retriever.retrieve("whole parent")

        assert "the whole parent passage" in outcome.contexts[0].text
        assert outcome.contexts[0].matched_text

    def test_matched_child_is_identifiable(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("parent about检索", "ch1")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("检索")

        context = outcome.contexts[0]
        assert context.matched_child
        assert context.matched_text

    def test_provenance_is_present(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("source tracked content", "Chapter Two")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("source tracked")

        context = outcome.contexts[0]
        assert context.source_name == "d1.pdf"
        assert context.breadcrumb == "Chapter Two"
        assert context.page == 1

    def test_oversized_parent_is_excerpted(self, settings, store):
        long_parent = "x" * 5000
        _populate(store, "d1", {"d1_p0": (long_parent, "ch1")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("x")

        assert len(outcome.contexts[0].text) <= settings.retrieval.parent_excerpt_chars + 10

    def test_total_budget_is_respected(self, settings, store):
        for index in range(6):
            _populate(
                store,
                f"d{index}",
                {f"d{index}_p0": ("shared budget content " * 60, "ch")},
                children_per_parent=1,
            )
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("shared budget content")

        total = sum(len(c.text) for c in outcome.contexts)
        assert total <= settings.retrieval.context_char_budget


class TestDiagnostics:
    def test_all_stages_are_inspectable(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("diagnostic content about search", "ch1")})
        retriever = _retriever(settings, store, FakeReranker())

        outcome = retriever.retrieve("diagnostic content")

        assert outcome.dense is not None
        assert outcome.sparse is not None
        assert outcome.fused
        assert outcome.reranked

    def test_channel_membership_is_visible(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("channel membership visible", "ch1")})
        retriever = _retriever(settings, store)

        outcome = retriever.retrieve("channel membership")

        channels = {ch for candidate in outcome.fused for ch in candidate.channels}
        assert channels <= {"dense", "sparse"}
        assert channels

    def test_rerank_scores_are_recorded(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("scored content", "ch1")})
        retriever = _retriever(settings, store, FakeReranker())

        outcome = retriever.retrieve("scored")

        assert any(c.rerank_score is not None for c in outcome.reranked)


class TestLexicalIndexConsistency:
    def test_rebuild_picks_up_new_documents(self, settings, store):
        retriever = _retriever(settings, store)
        assert retriever.retrieve("newly added phrase").empty

        _populate(store, "d1", {"d1_p0": ("newly added phrase here", "ch1")})
        assert retriever.rebuild_lexical_index() > 0

        assert retriever.retrieve("newly added phrase").sparse

    def test_rebuild_drops_removed_documents(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("temporary content", "ch1")})
        retriever = _retriever(settings, store)
        assert retriever.retrieve("temporary content").sparse

        store.delete_document("d1")
        retriever.rebuild_lexical_index()

        assert retriever.retrieve("temporary content").sparse == []

    def test_index_reflects_only_active_units(self, settings, store):
        _populate(store, "d1", {"d1_p0": ("inactive soon content", "ch1")})
        retriever = _retriever(settings, store)

        store.deactivate_document("d1")
        retriever.rebuild_lexical_index()

        assert retriever.retrieve("inactive soon").sparse == []

class TestEmptyKnowledgeBaseBehaviour:
    def test_empty_index_short_circuits_without_embedding(self, settings, store):
        """Regression: an empty knowledge base asked a question raised a
        configuration error from the embedding provider instead of answering that
        there is no content. With nothing ingested there is nothing to embed for.
        """

        class ExplodingEmbedder:
            calls = 0

            def embed_texts(self, texts):
                ExplodingEmbedder.calls += 1
                raise AssertionError("no embedding should be attempted when nothing is ingested")

        lexical = LexicalIndex()
        retriever = HybridRetriever(
            settings.retrieval, store, ExplodingEmbedder(), lexical, None
        )

        outcome = retriever.retrieve("anything at all")

        assert outcome.empty
        assert ExplodingEmbedder.calls == 0

    def test_stale_index_is_resynchronised_from_the_store(self, settings, store):
        """A populated store with an empty index must recover, not report nothing."""
        _populate(store, "d1", {"d1_p0": ("recoverable content about search", "ch1")})
        lexical = LexicalIndex()  # deliberately not rebuilt
        retriever = HybridRetriever(settings.retrieval, store, FakeEmbedder(), lexical, None)

        outcome = retriever.retrieve("recoverable content")

        assert outcome.sparse
        assert not outcome.empty
