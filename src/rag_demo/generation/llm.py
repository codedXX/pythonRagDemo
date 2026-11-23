"""Generation adapter for the OpenAI-compatible relay.

The relay exposes both ``/v1/chat/completions`` and ``/v1/messages`` (verified by
probing: both answer 401 without a key, while ``/v1/completions`` and
``/v1/rerank`` answer 404). We use the OpenAI-compatible route because it is the
one RAGAS-style tooling can also speak to later.

Streaming is a first-class requirement rather than an option: the interface
delivers answer text incrementally, and the self-check exercises the stream so a
non-streaming provider is discovered before a user hits it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import httpx

from rag_demo.config import LLMSettings
from rag_demo.errors import (
    ConfigurationIssue,
    ContractMismatch,
    TransientFailure,
    Unavailable,
)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class Message:
    """One conversational turn."""

    role: str
    content: str

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Generator:
    """Chat completions client with streaming support."""

    def __init__(self, settings: LLMSettings, sleep=time.sleep) -> None:
        self._settings = settings
        self._sleep = sleep

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    # ------------------------------------------------------------------ public

    def stream_chat(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield answer text incrementally as the provider produces it."""
        for chunk in self._stream(messages):
            yield chunk

    def complete(self, messages: Sequence[Message]) -> str:
        """Return the whole answer at once (used by query rewriting)."""
        return "".join(self.stream_chat(messages))

    def verify(self) -> str:
        """Prove the model exists, credentials work, and streaming streams."""
        pieces = list(
            self.stream_chat(
                [
                    Message(role="user", content="Reply with the single word: ok"),
                ]
            )
        )
        text = "".join(pieces)
        if not text.strip():
            raise ContractMismatch("llm", "self-check received an empty response")
        if len(pieces) < 1:  # pragma: no cover - defensive
            raise ContractMismatch("llm", "self-check observed no streamed chunks")
        return text.strip()

    # ----------------------------------------------------------------- internal

    def _endpoint(self) -> str:
        return self._settings.base_url.rstrip("/") + "/chat/completions"

    def _payload(self, messages: Sequence[Message], stream: bool) -> dict:
        payload = {
            "model": self._settings.model,
            "messages": [m.as_payload() for m in messages],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "stream": stream,
        }
        if stream:
            # Without this some gateways buffer the whole answer and emit it as
            # one chunk, which would defeat incremental delivery.
            payload["stream_options"] = {"include_usage": False}
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

    def _stream(self, messages: Sequence[Message]) -> Iterator[str]:
        if not self._settings.api_key:
            raise ConfigurationIssue(
                "llm",
                "no API key configured; set llm.api_key in config.yaml "
                "or RAG__LLM__API_KEY",
            )

        url = self._endpoint()
        attempts = 3
        for attempt in range(attempts):
            emitted = False
            try:
                with httpx.stream(
                    "POST",
                    url,
                    json=self._payload(messages, stream=True),
                    headers=self._headers(),
                    timeout=self._settings.timeout_seconds,
                ) as response:
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", "replace")
                        self._raise_for_status(response.status_code, body)

                    for line in response.iter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            return
                        piece = self._parse_chunk(data)
                        if piece:
                            emitted = True
                            yield piece
                return
            except TransientFailure:
                # Only retry when nothing has reached the caller yet; retrying
                # mid-stream would duplicate text in the answer.
                if emitted or attempt == attempts - 1:
                    raise
                self._sleep(1.0 * (2**attempt))

    @staticmethod
    def _parse_chunk(data: str) -> str:
        try:
            payload = json.loads(data)
        except ValueError as exc:
            raise ContractMismatch("llm", f"stream chunk was not JSON: {data[:120]}") from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        # Some providers put the text in "message" even while streaming.
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    def _raise_for_status(self, status: int, body: str) -> None:
        detail = self._extract_message(body)
        if status in _RETRYABLE_STATUS:
            raise TransientFailure("llm", f"HTTP {status}: {detail}", status=status)
        if status in {401, 403}:
            raise ConfigurationIssue(
                "llm",
                f"credentials rejected (HTTP {status}): {detail}. Check llm.api_key.",
            )
        if status == 404:
            raise ConfigurationIssue(
                "llm",
                f"endpoint not found (HTTP 404): {detail}. Check llm.base_url and "
                f"llm.model (currently '{self._settings.model}').",
            )
        if status == 400:
            raise ConfigurationIssue(
                "llm",
                f"request rejected (HTTP 400): {detail}. The configured model is "
                f"'{self._settings.model}'; the relay may not offer it.",
            )
        raise ContractMismatch("llm", f"unexpected HTTP {status}: {detail}")

    @staticmethod
    def _extract_message(body: str) -> str:
        try:
            payload = json.loads(body)
        except ValueError:
            return body[:300] or "(empty body)"
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if payload.get("message"):
                return str(payload["message"])
        return body[:300]

    def list_models(self) -> list[str]:
        """Return the models the relay advertises.

        Useful diagnostics when the configured model name is rejected: the relay
        exposes ``GET /v1/models`` (401 without a key, so it exists).
        """
        if not self._settings.api_key:
            raise ConfigurationIssue("llm", "no API key configured")
        url = self._settings.base_url.rstrip("/") + "/models"
        try:
            response = httpx.get(url, headers=self._headers(), timeout=30.0)
        except httpx.HTTPError as exc:
            raise Unavailable("llm", f"cannot reach {url}: {exc}") from exc
        if response.status_code != 200:
            raise ContractMismatch(
                "llm", f"model listing returned HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ContractMismatch("llm", "model listing was not JSON") from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise ContractMismatch("llm", "model listing has no 'data' array")
        return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]