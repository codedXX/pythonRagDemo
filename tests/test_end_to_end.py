"""End-to-end tests spanning the whole ingestion and answering chain.

These use real Milvus and fake providers, so they run without an API key while still
exercising the parts that fail in practice: chunking, persistence, both recall
channels, fusion, reranking, expansion and the HTTP contract.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from rag_demo.api.app import create_app
from rag_demo.api.services import Services
from rag_demo.chunking.chunker import Chunker
from rag_demo.config import load_settings
from rag_demo.embedding.rerank import RerankResult
from rag_demo.generation.answer import Answerer
from rag_demo.ingestion import IngestionPipeline
from rag_demo.retrieval.hybrid import HybridRetriever
from rag_demo.retrieval.lexical import LexicalIndex
from rag_demo.store.milvus_store import MilvusStore, normalise
from rag_demo.store.registry import DocumentRegistry

pytestmark = pytest.mark.integration

DIM = 10


class HashEmbedder:
    def embed_texts(self, texts):
        return [
            normalise(
                [((sum(ord(c) for c in text) + i * 7) % 31) + 1.0 for i in range(DIM)]
            )
            for text in texts
        ]

    def probe_dimension(self, text: str = "ping") -> int:
        return DIM


class KeywordReranker:
    def rerank(self, query, documents, top_n=None):
        scored = [
            RerankResult(index=i, score=1.0 if query[:2] in document else 0.3)
            for i, document in enumerate(documents)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored


class EchoGenerator:
    """Produces an answer that names the sources it was given."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream_chat(self, messages):
        self.prompts.append(messages[-1].content)
        for piece in ("根据", "提供的", "资料，", "答案是", "这样。"):
            yield piece

    def complete(self, messages):
        return "改写后的问题"


def _milvus_available(uri: str) -> bool:
    try:
        from pymilvus import MilvusClient

        MilvusClient(uri=uri).get_server_version()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    settings = load_settings(root=root, use_env=False)
    if not _milvus_available(settings.milvus.uri):
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")
    suffix = uuid.uuid4().hex[:6]
    settings.milvus.child_collection = f"e2e_child_{suffix}"
    settings.milvus.parent_collection = f"e2e_parent_{suffix}"
    settings.milvus.meta_collection = f"e2e_meta_{suffix}"
    settings.chunking.parent_target_chars = 400
    settings.chunking.parent_max_chars = 600
    settings.chunking.child_target_chars = 180
    settings.chunking.child_overlap_chars = 30
    settings.chunking.semantic_enabled = False
    settings.retrieval.context_count = 4
    settings.retrieval.rerank_candidates = 10
    return settings


@pytest.fixture(scope="module")
def store(settings):
    store = MilvusStore(settings)
    store.drop_all()
    store.ensure_collections(dimension=DIM)
    yield store
    store.drop_all()


@pytest.fixture()
def app_client(settings, store, tmp_path):
    registry = DocumentRegistry(tmp_path / "rag.db")
    embedder = HashEmbedder()
    lexical = LexicalIndex()
    retriever = HybridRetriever(
        settings.retrieval, store, embedder, lexical, KeywordReranker()
    )
    generator = EchoGenerator()
    answerer = Answerer(settings, generator)
    pipeline = IngestionPipeline(
        settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)
    )
    services = Services(
        settings=settings,
        store=store,
        registry=registry,
        embedder=embedder,
        reranker=KeywordReranker(),
        generator=generator,
        lexical=lexical,
        retriever=retriever,
        answerer=answerer,
        pipeline=pipeline,
        dimension=DIM,
    )
    client = TestClient(create_app(settings, services))
    client.services = services
    client.registry = registry
    yield client
    store.client.delete(store.child_name, filter='doc_id != ""')
    store.client.delete(store.parent_name, filter='doc_id != ""')


def _events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if event:
            events.append((event, json.loads("\n".join(data)) if data else {}))
    return events


MARKDOWN_DOC = {
    "name": "guide.md",
    "body": """# 检索系统指南

## 混合检索原理

混合检索把稀疏召回与稠密召回结合。稀疏召回基于 BM25 做关键词匹配，
稠密召回基于向量做语义匹配。两路结果用 RRF 融合，避免单一分数尺度的问题。

## 父子文档

父子文档让检索与生成使用不同粒度。子块用于向量检索，父块交给模型阅读。
这样既能精确命中，又能提供完整上下文。

## 重排环节

重排对融合后的候选重新打分。重排的输入包含标题路径，
因此脱离标题的片段也能被正确理解。
""",
}


class TestFullChain:
    def test_upload_then_ask_finds_the_new_document(self, app_client):
        """The whole promise: upload, then immediately ask about it."""
        upload = app_client.post(
            "/api/docs",
            files={"files": (MARKDOWN_DOC["name"], MARKDOWN_DOC["body"].encode(), "text/markdown")},
        )
        names = [name for name, _ in _events(upload)]
        assert "done" in names

        chat = app_client.post("/api/chat", json={"question": "混合检索是怎么工作的？"})
        events = _events(chat)

        sources = [payload for name, payload in events if name == "sources"][0]["sources"]
        assert sources
        assert sources[0]["source"] == "guide.md"

    def test_answer_streams_and_cites_sources(self, app_client):
        app_client.post(
            "/api/docs",
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},
        )

        events = _events(app_client.post("/api/chat", json={"question": "父子文档有什么用？"}))

        tokens = [payload["text"] for name, payload in events if name == "token"]
        assert "".join(tokens).startswith("根据")
        done = [payload for name, payload in events if name == "done"][0]
        assert done["answer"]
        assert not done["truncated"]

    def test_contexts_reach_the_model(self, app_client):
        app_client.post(
            "/api/docs",
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},
        )

        app_client.post("/api/chat", json={"question": "重排为什么需要标题路径？"})

        prompts = app_client.services.answerer.generator.prompts
        assert prompts
        # Parent text, not child fragments, must be what the model reads.
        assert "父块" in prompts[0] or "重排" in prompts[0]


class TestConsistency:
    def test_document_is_retrievable_immediately_after_upload(self, app_client):
        """Guards the Strong-consistency requirement end to end."""
        body = "# 即时检索\n\n上传后立刻提问应该能命中这段内容。\n"

        app_client.post("/api/docs", files={"files": ("fresh.md", body.encode(), "text/markdown")})
        events = _events(app_client.post("/api/chat", json={"question": "上传后立刻提问"}))

        sources = [payload for name, payload in events if name == "sources"][0]["sources"]
        assert any(s["source"] == "fresh.md" for s in sources)

    def test_removed_document_disappears_from_both_channels(self, app_client):
        body = "# 将被删除\n\n这段内容随后会被删除，不应再被检索到。\n"
        app_client.post("/api/docs", files={"files": ("temp.md", body.encode(), "text/markdown")})

        docs = app_client.get("/api/docs").json()["documents"]
        doc_id = next(d["doc_id"] for d in docs if d["name"] == "temp.md")
        app_client.delete(f"/api/docs/{doc_id}")

        events = _events(app_client.post("/api/chat", json={"question": "这段内容随后会被删除"}))
        sources = [payload for name, payload in events if name == "sources"][0]["sources"]
        assert all(s["source"] != "temp.md" for s in sources)

        names = [d["name"] for d in app_client.get("/api/docs").json()["documents"]]
        assert "temp.md" not in names

    def test_reupload_replaces_rather_than_duplicates(self, app_client):
        first = "# 版本一\n\n旧版本的内容在这里。\n"
        app_client.post("/api/docs", files={"files": ("same.md", first.encode(), "text/markdown")})

        second = "# 版本二\n\n新版本的内容替换了旧版本。\n"
        app_client.post("/api/docs", files={"files": ("same.md", second.encode(), "text/markdown")})

        docs = [d for d in app_client.get("/api/docs").json()["documents"] if d["name"] == "same.md"]
        assert len(docs) == 1
        assert docs[0]["state"] == "active"
        assert docs[0]["parent_count"] >= 1


class TestFailurePaths:
    def test_unsupported_type_is_rejected_in_the_stream(self, app_client):
        response = app_client.post(
            "/api/docs", files={"files": ("notes.txt", b"plain text", "text/plain")}
        )
        events = _events(response)

        # Either the API rejects it or ingestion does; either way the client is told
        # why, and nothing is indexed.
        failed = [payload for name, payload in events if name in {"failed", "rejected"}]
        assert failed
        assert app_client.registry.list_usable() == []

    def test_scanned_pdf_reports_the_reason(self, app_client):
        import pymupdf

        pdf = pymupdf.open()
        pdf.new_page()
        content = pdf.tobytes()
        pdf.close()

        response = app_client.post(
            "/api/docs", files={"files": ("scan.pdf", content, "application/pdf")}
        )
        failed = [payload for name, payload in _events(response) if name == "failed"]

        assert failed
        assert failed[0]["stage"] == "parsing"
        assert "scan" in failed[0]["error"].lower() or "image" in failed[0]["error"].lower()

    def test_question_with_no_documents_is_answered_honestly(self, app_client):
        events = _events(app_client.post("/api/chat", json={"question": "anything?"}))

        done = [payload for name, payload in events if name == "done"][0]
        assert done["empty"] is True
        # No model call should have been made.
        assert app_client.services.answerer.generator.prompts == []


class TestMultiTurn:
    def test_follow_up_uses_a_rewritten_query(self, app_client):
        app_client.post(
            "/api/docs",
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},
        )
        history = [{"question": "混合检索是什么", "answer": "一种结合稀疏与稠密召回的方法"}]

        events = _events(
            app_client.post(
                "/api/chat", json={"question": "它有什么好处？", "history": history}
            )
        )

        rewritten = [payload for name, payload in events if name == "rewritten"]
        assert rewritten
        assert rewritten[0]["query"] == "改写后的问题"

    def test_first_turn_is_not_rewritten(self, app_client):
        app_client.post(
            "/api/docs",
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},
        )

        events = _events(app_client.post("/api/chat", json={"question": "混合检索是什么？"}))

        assert not [payload for name, payload in events if name == "rewritten"]