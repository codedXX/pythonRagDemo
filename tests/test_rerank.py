"""Tests for the native DashScope rerank adapter.

The adapter must not behave like the embedding one: this endpoint has a different
request body, a different response envelope and a different error shape.
"""

from __future__ import annotations

import httpx
import pytest

from rag_demo.config import RerankSettings
from rag_demo.embedding.rerank import RerankResult, Reranker
from rag_demo.errors import ConfigurationIssue, ContractMismatch, TransientFailure

URL = "https://dashscope.example.com/api/v1/services/rerank/text-rerank/text-rerank"


def _settings(**overrides) -> RerankSettings:
    base = dict(
        base_url="https://dashscope.example.com",
        path="/api/v1/services/rerank/text-rerank/text-rerank",
        api_key="sk-test",
        model="test-rerank",
        max_retries=3,
    )
    base.update(overrides)
    return RerankSettings(**base)


def _ok(results: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"output": {"results": results}, "usage": {"total_tokens": 12}},
        request=httpx.Request("POST", URL),
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("unexpected network call in test")

    monkeypatch.setattr(httpx, "post", explode)


def test_url_is_the_native_path_not_the_compatible_one():
    reranker = Reranker(_settings())

    # Guard against the mistake this whole module exists to avoid.
    assert reranker.settings.url == URL
    assert "/compatible-mode/" not in reranker.settings.url


def test_parses_native_envelope_and_sorts_by_score(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _ok(
            [
                {"index": 0, "relevance_score": 0.10},
                {"index": 2, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.42},
            ]
        ),
    )

    results = reranker.rerank("q", ["a", "b", "c"])

    assert [r.index for r in results] == [2, 1, 0]
    assert results[0].score == pytest.approx(0.95)


def test_sends_the_native_request_shape(monkeypatch):
    captured: dict = {}

    def capture(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _ok([{"index": 0, "relevance_score": 0.5}])

    monkeypatch.setattr(httpx, "post", capture)
    Reranker(_settings()).rerank("query text", ["doc one", "doc two"])

    assert captured["url"] == URL
    # The native contract nests query and documents under "input"; sending the
    # OpenAI shape here would be rejected.
    assert captured["json"]["input"] == {
        "query": "query text",
        "documents": ["doc one", "doc two"],
    }
    assert captured["json"]["model"] == "test-rerank"


def test_rejects_openai_shaped_response(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            200,
            json={"data": [{"index": 0, "relevance_score": 0.5}]},
            request=httpx.Request("POST", URL),
        ),
    )

    with pytest.raises(ContractMismatch) as excinfo:
        reranker.rerank("q", ["a"])

    assert "output" in str(excinfo.value)


def test_rejects_out_of_range_index(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _ok([{"index": 7, "relevance_score": 0.9}])
    )

    with pytest.raises(ContractMismatch):
        reranker.rerank("q", ["a"])


def test_rejects_duplicate_indices(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _ok(
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
            ]
        ),
    )

    with pytest.raises(ContractMismatch):
        reranker.rerank("q", ["a", "b"])


def test_rejects_non_numeric_score(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _ok([{"index": 0, "relevance_score": "high"}])
    )

    with pytest.raises(ContractMismatch):
        reranker.rerank("q", ["a"])


def test_404_explains_the_compatible_path_trap(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(404, json={}, request=httpx.Request("POST", URL)),
    )

    with pytest.raises(ConfigurationIssue) as excinfo:
        reranker.rerank("q", ["a"])

    message = str(excinfo.value)
    assert "native path" in message
    assert "compatible" in message


def test_invalid_key_uses_dashscope_error_shape(monkeypatch):
    reranker = Reranker(_settings())
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(
            401,
            json={"code": "InvalidApiKey", "message": "Invalid API-key provided."},
            request=httpx.Request("POST", URL),
        ),
    )

    with pytest.raises(ConfigurationIssue) as excinfo:
        reranker.rerank("q", ["a"])

    # The DashScope code should survive into the message; the OpenAI-shaped
    # parser would have produced "(empty body)" here.
    assert "InvalidApiKey" in str(excinfo.value)


def test_missing_api_key_is_configuration_issue():
    with pytest.raises(ConfigurationIssue):
        Reranker(_settings(api_key="")).rerank("q", ["a"])


def test_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def flaky(url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={}, request=httpx.Request("POST", URL))
        return _ok([{"index": 0, "relevance_score": 0.8}])

    monkeypatch.setattr(httpx, "post", flaky)
    results = Reranker(_settings(), sleep=lambda s: None).rerank("q", ["a"])

    assert results[0].index == 0
    assert attempts["n"] == 2


def test_transient_exhaustion_raises(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: httpx.Response(503, json={}, request=httpx.Request("POST", URL)),
    )

    with pytest.raises(TransientFailure):
        Reranker(_settings(max_retries=2), sleep=lambda s: None).rerank("q", ["a"])


def test_empty_documents_makes_no_request(monkeypatch):
    assert Reranker(_settings()).rerank("q", []) == []


def test_verify_accepts_sensible_ranking(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _ok(
            [
                {"index": 0, "relevance_score": 0.91},
                {"index": 1, "relevance_score": 0.05},
                {"index": 2, "relevance_score": 0.02},
            ]
        ),
    )

    results = Reranker(_settings()).verify()

    assert results[0].index == 0


def test_verify_rejects_implausible_ranking(monkeypatch):
    # Responds correctly but ranks the irrelevant passage first: the endpoint is
    # reachable yet unusable, which reachability alone would not catch.
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _ok(
            [
                {"index": 2, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.01},
            ]
        ),
    )

    with pytest.raises(ContractMismatch) as excinfo:
        Reranker(_settings()).verify()

    assert "implausible" in str(excinfo.value)


def test_top_n_is_sent_when_requested(monkeypatch):
    captured: dict = {}

    def capture(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _ok([{"index": 0, "relevance_score": 0.5}])

    monkeypatch.setattr(httpx, "post", capture)
    Reranker(_settings()).rerank("q", ["a", "b"], top_n=1)

    assert captured["json"]["parameters"] == {"top_n": 1}