"""Tests for the generation adapter: streaming, error mapping, sanity checks."""

from __future__ import annotations

import json

import httpx
import pytest

from rag_demo.config import LLMSettings
from rag_demo.errors import ConfigurationIssue, ContractMismatch, TransientFailure
from rag_demo.generation.llm import Generator, Message

URL = "https://relay.example.com/v1/chat/completions"


def _settings(**overrides) -> LLMSettings:
    base = dict(
        base_url="https://relay.example.com/v1",
        api_key="sk-test",
        model="claude-sonnet-5",
        timeout_seconds=5.0,
    )
    base.update(overrides)
    return LLMSettings(**base)


class _FakeStreamResponse:
    """Minimal stand-in for httpx's streaming response context manager."""

    def __init__(self, status_code: int, lines: list[str], body: str = "") -> None:
        self.status_code = status_code
        self._lines = lines
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return self._body.encode()


def _sse(pieces: list[str]) -> list[str]:
    lines = []
    for piece in pieces:
        lines.append(
            "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
        )
        lines.append("")
    lines.append("data: [DONE]")
    return lines


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("unexpected network call in test")

    monkeypatch.setattr(httpx, "stream", explode)
    monkeypatch.setattr(httpx, "get", explode)


def test_streams_pieces_incrementally(monkeypatch):
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["Hel", "lo", " world"]))
    )

    pieces = list(Generator(_settings()).stream_chat([Message("user", "hi")]))

    assert pieces == ["Hel", "lo", " world"]
    assert "".join(pieces) == "Hello world"


def test_stream_request_asks_for_streaming(monkeypatch):
    captured: dict = {}

    def capture(method, url, json=None, headers=None, timeout=None):
        captured["json"] = json
        captured["url"] = url
        return _FakeStreamResponse(200, _sse(["ok"]))

    monkeypatch.setattr(httpx, "stream", capture)
    list(Generator(_settings()).stream_chat([Message("user", "hi")]))

    assert captured["url"] == URL
    assert captured["json"]["stream"] is True
    assert captured["json"]["stream_options"] == {"include_usage": False}
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_joins_chunks(monkeypatch):
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["a", "b"]))
    )

    assert Generator(_settings()).complete([Message("user", "x")]) == "ab"


def test_tolerates_message_shaped_chunks(monkeypatch):
    lines = [
        "data: " + json.dumps({"choices": [{"message": {"content": "alt"}}]}),
        "data: [DONE]",
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, lines))

    assert Generator(_settings()).complete([Message("user", "x")]) == "alt"


def test_skips_non_data_lines(monkeypatch):
    lines = [": keep-alive", "", "data: " + json.dumps({"choices": []}), "data: [DONE]"]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, lines))

    assert Generator(_settings()).complete([Message("user", "x")]) == ""


def test_missing_api_key_is_configuration_issue():
    with pytest.raises(ConfigurationIssue):
        list(Generator(_settings(api_key="")).stream_chat([Message("user", "x")]))


def test_401_is_configuration_issue(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: _FakeStreamResponse(
            401, [], json.dumps({"error": {"message": "Incorrect API key provided."}})
        ),
    )

    with pytest.raises(ConfigurationIssue) as excinfo:
        list(Generator(_settings()).stream_chat([Message("user", "x")]))

    assert "credentials" in str(excinfo.value)


def test_unknown_model_names_the_model(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: _FakeStreamResponse(
            400, [], json.dumps({"error": {"message": "model not found"}})
        ),
    )

    with pytest.raises(ConfigurationIssue) as excinfo:
        list(Generator(_settings(model="claude-sonnet-5")).stream_chat([Message("user", "x")]))

    assert "claude-sonnet-5" in str(excinfo.value)


def test_retry_when_nothing_emitted_yet(monkeypatch):
    attempts = {"n": 0}

    def flaky(method, url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FakeStreamResponse(503, [], "")
        return _FakeStreamResponse(200, _sse(["ok"]))

    monkeypatch.setattr(httpx, "stream", flaky)

    text = Generator(_settings(), sleep=lambda s: None).complete([Message("user", "x")])

    assert text == "ok"
    assert attempts["n"] == 2


def test_does_not_retry_after_emitting(monkeypatch):
    """Retrying mid-stream would duplicate text in the answer."""
    attempts = {"n": 0}

    class FailsAfterOneChunk(_FakeStreamResponse):
        def iter_lines(self):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "partial"}}]})
            raise TransientFailure("llm", "connection dropped")

    def flaky(method, url, json=None, headers=None, timeout=None):
        attempts["n"] += 1
        return FailsAfterOneChunk(200, [])

    monkeypatch.setattr(httpx, "stream", flaky)
    seen: list[str] = []

    with pytest.raises(TransientFailure):
        for piece in Generator(_settings(), sleep=lambda s: None).stream_chat([Message("user", "x")]):
            seen.append(piece)

    assert seen == ["partial"]
    assert attempts["n"] == 1


def test_bad_json_chunk_is_contract_mismatch(monkeypatch):
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, ["data: {not json"])
    )

    with pytest.raises(ContractMismatch):
        list(Generator(_settings()).stream_chat([Message("user", "x")]))


def test_verify_rejects_empty_answer(monkeypatch):
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["   "]))
    )

    with pytest.raises(ContractMismatch):
        Generator(_settings()).verify()


def test_verify_returns_text(monkeypatch):
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["ok"]))
    )

    assert Generator(_settings()).verify() == "ok"


def test_list_models_returns_ids(monkeypatch):
    response = httpx.Response(
        200,
        json={"data": [{"id": "claude-sonnet-5"}, {"id": "gpt-4o"}]},
        request=httpx.Request("GET", "https://relay.example.com/v1/models"),
    )
    monkeypatch.setattr(httpx, "get", lambda *a, **k: response)

    assert Generator(_settings()).list_models() == ["claude-sonnet-5", "gpt-4o"]