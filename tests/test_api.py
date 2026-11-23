"""Tests for the HTTP API, using injected fakes.

No external service is contacted: these tests are about the transport contract — SSE
event ordering, error mapping, and the promise that no secret reaches the client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from rag_demo.api.app import create_app
from rag_demo.api.services import Services
from rag_demo.config import Settings
from rag_demo.retrieval.hybrid import Context, RetrievalOutcome
from rag_demo.retrieval.lexical import LexicalIndex
from rag_demo.selfcheck import DependencyReport, SelfCheckReport, Status
from rag_demo.store.registry import DocumentRecord, DocumentState


class FakeRegistry:
    def __init__(self) -> None:
        self.records: dict[str, DocumentRecord] = {}
        self.removed: list[str] = []

    def list_documents(self):
        return list(self.records.values())

    def list_usable(self):
        return [r for r in self.records.values() if r.state is DocumentState.ACTIVE]

    def get(self, doc_id):
        return self.records.get(doc_id)

    def remove(self, doc_id):
        self.records.pop(doc_id, None)
        self.removed.append(doc_id)

    def add(self, doc_id: str, name: str, state=DocumentState.ACTIVE):
        self.records[doc_id] = DocumentRecord(
            doc_id=doc_id, name=name, state=state, parent_count=2, child_count=4
        )


class FakeRetriever:
    def __init__(self, contexts=None, error=None) -> None:
        self.contexts = contexts if contexts is not None else [_default_context()]
        self.error = error
        self.queries: list[str] = []

    def retrieve(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return RetrievalOutcome(query=query, contexts=self.contexts)

    def rebuild_lexical_index(self):
        return len(self.contexts)


class FakeAnswerer:
    def __init__(self, pieces=("Hello", " world"), rewrite=None, failing=False) -> None:
        self.pieces = pieces
        self.rewrite_result = rewrite
        self.failing = failing
        self.questions: list[str] = []

    def rewrite(self, question, conversation):
        if self.rewrite_result:
            return self.rewrite_result, True
        return question, False

    def stream_answer(self, question, contexts, conversation=None):
        self.questions.append(question)
        if self.failing:
            raise RuntimeError("generator exploded")
        yield from self.pieces


class FakePipeline:
    def __init__(self, error=None) -> None:
        self.error = error
        self.ingested: list[tuple[str, bytes]] = []

    def ingest(self, name, content, on_progress=None, doc_id=None):
        from rag_demo.ingestion import IngestResult, IngestionError, Progress, Stage

        self.ingested.append((name, content))
        for stage in (Stage.PARSING, Stage.CHUNKING, Stage.EMBEDDING, Stage.PERSISTING):
            if on_progress:
                on_progress(Progress(stage=stage, message=f"{stage.value}...", current=1, total=2))
        if self.error:
            raise self.error
        return IngestResult(
            doc_id="d1", name=name, parent_count=2, child_count=4, char_count=500, duration_seconds=0.4
        )


class FakeStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_document(self, doc_id):
        self.deleted.append(doc_id)


def _default_context() -> Context:
    return Context(
        parent_id="p1",
        text="the retrieved passage",
        breadcrumb="Chapter One",
        page=4,
        source_name="paper.pdf",
        matched_child="p1_c0",
        matched_text="the matched fragment",
        rerank_score=0.87,
    )


@dataclass
class Fakes:
    registry: FakeRegistry = field(default_factory=FakeRegistry)
    retriever: FakeRetriever = field(default_factory=FakeRetriever)
    answerer: FakeAnswerer = field(default_factory=FakeAnswerer)
    pipeline: FakePipeline = field(default_factory=FakePipeline)
    store: FakeStore = field(default_factory=FakeStore)
    lexical: LexicalIndex = field(default_factory=LexicalIndex)


def _client(fakes: Fakes, health: SelfCheckReport | None = None) -> TestClient:
    settings = Settings()
    services = Services(
        settings=settings,
        store=fakes.store,  # type: ignore[arg-type]
        registry=fakes.registry,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        reranker=None,  # type: ignore[arg-type]
        generator=None,  # type: ignore[arg-type]
        lexical=fakes.lexical,
        retriever=fakes.retriever,  # type: ignore[arg-type]
        answerer=fakes.answerer,  # type: ignore[arg-type]
        pipeline=fakes.pipeline,  # type: ignore[arg-type]
        selfcheck=health,
        dimension=1024,
    )
    app = create_app(settings, services)
    return TestClient(app)


def _events(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, payload) pairs."""
    events: list[tuple[str, dict]] = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data.append(line[len("data: ") :])
        if event:
            payload = json.loads("\n".join(data)) if data else {}
            events.append((event, payload))
    return events


class TestHealth:
    def test_reports_dependency_status(self):
        health = SelfCheckReport(
            dependencies=[
                DependencyReport("embedding", Status.HEALTHY, "ok", {"dimension": 1024}),
                DependencyReport("rerank", Status.MISCONFIGURED, "no key"),
            ],
            embedding_dimension=1024,
        )
        client = _client(Fakes(), health)

        body = client.get("/api/health").json()

        assert body["healthy"] is False
        assert body["dimension"] == 1024
        by_name = {d["name"]: d["status"] for d in body["dependencies"]}
        assert by_name["embedding"] == "healthy"
        assert by_name["rerank"] == "misconfigured"

    def test_never_exposes_secrets(self):
        settings = Settings()
        settings.embedding.api_key = "sk-super-secret-value"
        settings.llm.api_key = "sk-relay-secret-value"
        settings.rerank.api_key = "sk-rerank-secret-value"
        fakes = Fakes()
        services = Services(
            settings=settings,
            store=fakes.store,
            registry=fakes.registry,
            embedder=None,
            reranker=None,
            generator=None,
            lexical=fakes.lexical,
            retriever=fakes.retriever,
            answerer=fakes.answerer,
            pipeline=fakes.pipeline,
            dimension=1024,
        )

        response = create_app(settings, services)
        body = TestClient(response).get("/api/health").text

        assert "sk-super-secret-value" not in body
        assert "sk-relay-secret-value" not in body
        assert "sk-rerank-secret-value" not in body


class TestChatStream:
    def test_sources_arrive_before_any_token(self):
        """The ordering users depend on: sources first, tokens after."""
        client = _client(Fakes())

        events = _events(client.post("/api/chat", json={"question": "what?"}))

        names = [name for name, _ in events]
        assert "sources" in names
        assert names.index("sources") < names.index("token")

    def test_tokens_are_streamed_individually(self):
        client = _client(Fakes())

        events = _events(client.post("/api/chat", json={"question": "what?"}))

        tokens = [payload["text"] for name, payload in events if name == "token"]
        assert tokens == ["Hello", " world"]

    def test_done_carries_the_full_answer(self):
        client = _client(Fakes())

        events = _events(client.post("/api/chat", json={"question": "what?"}))

        done = [payload for name, payload in events if name == "done"][0]
        assert done["answer"] == "Hello world"
        assert done["truncated"] is False

    def test_sources_include_provenance(self):
        client = _client(Fakes())

        events = _events(client.post("/api/chat", json={"question": "what?"}))
        sources = [payload for name, payload in events if name == "sources"][0]["sources"]

        assert sources[0]["source"] == "paper.pdf"
        assert sources[0]["page"] == 4
        assert sources[0]["breadcrumb"] == "Chapter One"
        assert sources[0]["score"] == pytest.approx(0.87)

    def test_status_events_bracket_the_phases(self):
        client = _client(Fakes())

        events = _events(client.post("/api/chat", json={"question": "what?"}))
        stages = [payload["stage"] for name, payload in events if name == "status"]

        assert stages == ["rewriting", "retrieving"]

    def test_rewritten_query_is_reported(self):
        fakes = Fakes(answerer=FakeAnswerer(rewrite="a standalone question"))
        client = _client(fakes)

        events = _events(client.post("/api/chat", json={"question": "it?"}))

        rewritten = [payload for name, payload in events if name == "rewritten"]
        assert rewritten[0]["query"] == "a standalone question"
        assert fakes.retriever.queries == ["a standalone question"]

    def test_empty_knowledge_base_is_explained(self):
        fakes = Fakes(retriever=FakeRetriever(contexts=[]))
        client = _client(fakes)

        events = _events(client.post("/api/chat", json={"question": "anything?"}))

        done = [payload for name, payload in events if name == "done"][0]
        assert done["empty"] is True
        # Generation must not have been attempted.
        assert fakes.answerer.questions == []

    def test_mid_stream_failure_is_reported_as_truncated(self):
        fakes = Fakes(answerer=FakeAnswerer(failing=True))
        client = _client(fakes)

        events = _events(client.post("/api/chat", json={"question": "what?"}))

        errors = [payload for name, payload in events if name == "error"]
        assert errors and errors[0]["truncated"] is True
        done = [payload for name, payload in events if name == "done"][0]
        assert done["truncated"] is True

    def test_retrieval_failure_is_reported_without_tokens(self):
        fakes = Fakes(retriever=FakeRetriever(error=RuntimeError("dense channel down")))
        client = _client(fakes)

        events = _events(client.post("/api/chat", json={"question": "what?"}))

        names = [name for name, _ in events]
        assert "error" in names
        assert "token" not in names

    def test_missing_question_is_rejected(self):
        client = _client(Fakes())

        response = client.post("/api/chat", json={})

        assert response.status_code == 400

    def test_history_is_accepted(self):
        client = _client(Fakes())

        response = client.post(
            "/api/chat",
            json={
                "question": "follow up",
                "history": [{"question": "first", "answer": "first answer"}],
            },
        )

        assert response.status_code == 200

    def test_debug_event_is_opt_in(self):
        client = _client(Fakes())

        without = _events(client.post("/api/chat", json={"question": "q"}))
        with_debug = _events(client.post("/api/chat", json={"question": "q", "debug": True}))

        assert "debug" not in [name for name, _ in without]
        assert "debug" in [name for name, _ in with_debug]

    def test_debug_reports_every_stage(self):
        client = _client(Fakes())

        events = _events(client.post("/api/chat", json={"question": "q", "debug": True}))
        debug = [payload for name, payload in events if name == "debug"][0]

        assert set(debug) >= {"query", "dense", "sparse", "fused", "reranked"}


class TestUploadStream:
    def test_progress_events_precede_completion(self):
        client = _client(Fakes())

        response = client.post(
            "/api/docs", files={"files": ("notes.md", b"# hi\n\ntext", "text/markdown")}
        )
        names = [name for name, _ in _events(response)]

        assert names[0] == "start"
        assert "progress" in names
        assert names.index("progress") < names.index("done")
        assert names[-1] == "complete"

    def test_stages_are_reported_in_order(self):
        client = _client(Fakes())

        response = client.post(
            "/api/docs", files={"files": ("notes.md", b"content", "text/markdown")}
        )
        stages = [
            payload["stage"]
            for name, payload in _events(response)
            if name == "progress"
        ]

        assert stages == ["parsing", "chunking", "embedding", "persisting"]

    def test_failure_names_the_stage(self):
        from rag_demo.ingestion import IngestionError, Stage

        fakes = Fakes(pipeline=FakePipeline(error=IngestionError(Stage.PARSING, "scanned pdf")))
        client = _client(fakes)

        response = client.post(
            "/api/docs", files={"files": ("scan.pdf", b"%PDF-1.4", "application/pdf")}
        )
        failed = [payload for name, payload in _events(response) if name == "failed"]

        assert failed[0]["stage"] == "parsing"
        assert "scanned" in failed[0]["error"]

    def test_oversized_file_is_rejected_before_ingestion(self):
        fakes = Fakes()
        client = _client(fakes)
        # The limit is expressed in MB on settings; a 1-byte limit is not settable,
        # so this exercises the rejection path with a patched limit.
        client.app.state.settings.storage.max_upload_mb = 0

        response = client.post(
            "/api/docs", files={"files": ("big.md", b"x" * 10, "text/markdown")}
        )
        names = [name for name, _ in _events(response)]

        assert "rejected" in names
        assert fakes.pipeline.ingested == []


class TestDocumentManagement:
    def test_listing_returns_documents_with_state(self):
        fakes = Fakes()
        fakes.registry.add("d1", "paper.pdf")
        fakes.registry.add("d2", "broken.pdf", state=DocumentState.FAILED)
        client = _client(fakes)

        body = client.get("/api/docs").json()

        states = {d["name"]: d["state"] for d in body["documents"]}
        assert states == {"paper.pdf": "active", "broken.pdf": "failed"}

    def test_deleting_removes_content_and_returns_id(self):
        fakes = Fakes()
        fakes.registry.add("d1", "paper.pdf")
        client = _client(fakes)

        response = client.delete("/api/docs/d1")

        assert response.json()["removed"] == "d1"
        assert fakes.store.deleted == ["d1"]

    def test_deleting_unknown_document_is_404(self):
        client = _client(Fakes())

        assert client.delete("/api/docs/nope").status_code == 404


def test_index_page_is_served(monkeypatch, tmp_path):
    """The page is served from the static directory by the same process."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html><body>rag</body></html>", encoding="utf-8")
    monkeypatch.setattr("rag_demo.api.app.STATIC_DIR", static)

    client = _client(Fakes())

    response = client.get("/")

    assert response.status_code == 200
    assert "rag" in response.text


def test_index_contains_no_secrets(monkeypatch, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html><body>page</body></html>", encoding="utf-8")
    monkeypatch.setattr("rag_demo.api.app.STATIC_DIR", static)
    settings = Settings()
    settings.llm.api_key = "sk-must-not-leak"
    fakes = Fakes()
    services = Services(
        settings=settings,
        store=fakes.store,
        registry=fakes.registry,
        embedder=None,
        reranker=None,
        generator=None,
        lexical=fakes.lexical,
        retriever=fakes.retriever,
        answerer=fakes.answerer,
        pipeline=fakes.pipeline,
        dimension=1024,
    )
    client = TestClient(create_app(settings, services))

    body = client.get("/").text

    assert "sk-must-not-leak" not in body

def test_retrieval_error_is_not_described_as_a_generation_error():
    """The two failures send the user to different places, so the wording differs."""
    from rag_demo.api.routes import describe_retrieval_error
    from rag_demo.errors import ConfigurationIssue

    message = describe_retrieval_error(ConfigurationIssue("embedding", "no API key"))

    assert "Retrieval" in message
    assert "Generation" not in message
    assert "embedding" in message.lower() or "no API key" in message
