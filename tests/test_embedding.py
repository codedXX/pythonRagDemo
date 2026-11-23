"""Tests for the embedding adapter: parsing, error mapping, batching, caching."""

from __future__ import annotations

import httpx
import pytest

from rag_demo.config import EmbeddingSettings
from rag_demo.embedding.provider import (
    EmbeddingCache,
    EmbeddingClient,
    Embedder,
    content_hash,
)
from rag_demo.errors import ConfigurationIssue, ContractMismatch, TransientFailure

DIM = 8


def _settings(**overrides) -> EmbeddingSettings:
    base = dict(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="test-embedding",
        batch_size=2,
        max_retries=3,
        retry_base_delay_seconds=0.0,
    )
    base.update(overrides)
    return EmbeddingSettings(**base)


def _reply(vectors: list[list[float]], indices: list[int] | None = None) -> httpx.Response:
    order = indices if indices is not None else list(range(len(vectors)))
    payload = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in zip(order, vectors, strict=True)
        ],
    }
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://x"))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if a test accidentally performs a real request."""

    def explode(*args, **kwargs):
        raise AssertionError("unexpected network call in test")

    monkeypatch.setattr(httpx, "post", explode)


def test_parse_returns_vectors_in_input_order(monkeypatch):
    client = EmbeddingClient(_settings())
    # The provider returned them out of order; 'index' must be honoured,
    # otherwise vectors would be attached to the wrong chunks.
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _reply([[1.0] * DIM, [2.0] * DIM], indices=[1, 0])
    )

    vectors = client.embed(["first", "second"])

    assert vectors[0] == [2.0] * DIM
    assert vectors[1] == [1.0] * DIM


def test_parse_rejects_length_mismatch(monkeypatch):
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply([[1.0] * DIM]))

    with pytest.raises(ContractMismatch):
        client.embed(["a", "b"])


def test_parse_rejects_missing_data(monkeypatch):
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(200, json={"nope": True}, request=httpx.Request("POST", "https://x")),
    )

    with pytest.raises(ContractMismatch):
        client.embed(["a"])


def test_parse_rejects_inconsistent_dimensions(monkeypatch):
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _reply([[1.0] * DIM, [1.0] * (DIM + 1)])
    )

    with pytest.raises(ContractMismatch):
        client.embed(["a", "b"])


def test_missing_api_key_is_a_configuration_issue():
    client = EmbeddingClient(_settings(api_key=""))

    with pytest.raises(ConfigurationIssue) as excinfo:
        client.embed(["a"])

    assert "api_key" in str(excinfo.value)


def test_401_is_configuration_not_transient(monkeypatch):
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            401,
            json={"error": {"message": "Incorrect API key provided."}},
            request=httpx.Request("POST", "https://x"),
        ),
    )

    with pytest.raises(ConfigurationIssue) as excinfo:
        client.embed(["a"])

    assert "credentials" in str(excinfo.value)


def test_unknown_model_400_is_configuration(monkeypatch):
    client = EmbeddingClient(_settings(model="klingon-embedding-9"))
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            400,
            json={"error": {"message": "The model does not exist."}},
            request=httpx.Request("POST", "https://x"),
        ),
    )

    with pytest.raises(ConfigurationIssue) as excinfo:
        client.embed(["a"])

    # The message must name the model so the fix is a one-line config edit.
    assert "klingon-embedding-9" in str(excinfo.value)


def test_429_is_transient(monkeypatch):
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: httpx.Response(429, json={}, request=httpx.Request("POST", "https://x"))
    )

    with pytest.raises(TransientFailure):
        client.embed(["a"])


def test_unreachable_is_reported_as_unavailable(monkeypatch):
    client = EmbeddingClient(_settings())

    def boom(*args, **kwargs):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "post", boom)

    with pytest.raises(Exception) as excinfo:
        client.embed(["a"])

    assert "cannot reach" in str(excinfo.value)


def test_embedder_batches_requests(monkeypatch):
    client = EmbeddingClient(_settings(batch_size=2))
    calls: list[int] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(len(json["input"]))
        return _reply([[0.5] * DIM for _ in json["input"]])

    monkeypatch.setattr(httpx, "post", fake_post)
    embedder = Embedder(_settings(batch_size=2), client=client)

    vectors = embedder.embed_texts(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert calls == [2, 2, 1]


def test_embedder_retries_transient_then_succeeds(monkeypatch):
    client = EmbeddingClient(_settings())
    attempts = {"n": 0}

    def flaky(url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={}, request=httpx.Request("POST", "https://x"))
        return _reply([[0.1] * DIM for _ in json["input"]])

    monkeypatch.setattr(httpx, "post", flaky)
    embedder = Embedder(_settings(max_retries=3), client=client, sleep=lambda s: None)

    vectors = embedder.embed_texts(["a"])

    assert len(vectors) == 1
    assert attempts["n"] == 2


def test_embedder_does_not_retry_configuration_errors(monkeypatch):
    client = EmbeddingClient(_settings())
    attempts = {"n": 0}

    def deny(url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        return httpx.Response(
            401,
            json={"error": {"message": "Incorrect API key provided."}},
            request=httpx.Request("POST", "https://x"),
        )

    monkeypatch.setattr(httpx, "post", deny)
    embedder = Embedder(_settings(max_retries=5), client=client, sleep=lambda s: None)

    with pytest.raises(ConfigurationIssue):
        embedder.embed_texts(["a"])

    # Retrying a wrong key five times would waste time and quota.
    assert attempts["n"] == 1


def test_cache_avoids_second_request(monkeypatch, tmp_path):
    client = EmbeddingClient(_settings())
    calls = {"n": 0}

    def counting(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _reply([[0.7] * DIM for _ in json["input"]])

    monkeypatch.setattr(httpx, "post", counting)
    cache = EmbeddingCache(tmp_path)
    embedder = Embedder(_settings(), client=client, cache=cache)

    first = embedder.embed_texts(["hello"])
    second = embedder.embed_texts(["hello"])

    assert first == second
    assert calls["n"] == 1


def test_cache_miss_when_model_changes(monkeypatch, tmp_path):
    calls = {"n": 0}

    def counting(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        return _reply([[0.7] * DIM for _ in json["input"]])

    monkeypatch.setattr(httpx, "post", counting)
    cache = EmbeddingCache(tmp_path)

    Embedder(_settings(model="model-a"), client=EmbeddingClient(_settings(model="model-a")), cache=cache).embed_texts(["hi"])
    Embedder(_settings(model="model-b"), client=EmbeddingClient(_settings(model="model-b")), cache=cache).embed_texts(["hi"])

    # Vectors from different models live in different spaces; reusing them
    # silently would corrupt retrieval.
    assert calls["n"] == 2


def test_content_hash_differs_per_model_and_text():
    assert content_hash("a", "m") != content_hash("a", "n") != content_hash("b", "m")
    assert content_hash("a", "m") == content_hash("a", "m")


def test_probe_dimension_reports_vector_length(monkeypatch):
    client = EmbeddingClient(_settings())
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply([[0.0] * 1536]))

    assert Embedder(_settings(), client=client).probe_dimension() == 1536