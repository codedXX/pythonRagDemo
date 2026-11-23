"""Reranking adapter for Alibaba Cloud Bailian.

This is deliberately *not* an OpenAI-compatible client. Bailian's reranking is
served only on the native DashScope path:

    POST /api/v1/services/rerank/text-rerank/text-rerank
    {"model": ..., "input": {"query": ..., "documents": [...]}}
    -> {"output": {"results": [{"index": 0, "relevance_score": 0.93}, ...]}}

The OpenAI-compatible path ``/compatible-mode/v1/rerank`` returns HTTP 404
(verified against the live service), and the error envelope differs too: this
endpoint replies ``{"code": "...", "message": "..."}`` rather than OpenAI's
``{"error": {"message": ...}}``. Sharing one client with the embedding adapter
would fail in a confusing way, so the protocols are kept separate on purpose.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from rag_demo.config import RerankSettings
from rag_demo.errors import (
    ConfigurationIssue,
    ContractMismatch,
    TransientFailure,
    Unavailable,
)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RerankResult:
    """One reranked document: its original position and the model's score."""

    __slots__ = ("index", "score")

    def __init__(self, index: int, score: float) -> None:
        self.index = index
        self.score = score

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RerankResult(index={self.index}, score={self.score:.4f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RerankResult):
            return NotImplemented
        return self.index == other.index and self.score == other.score


class Reranker:
    """Client for the native DashScope text-rerank endpoint."""

    def __init__(self, settings: RerankSettings, sleep=time.sleep) -> None:
        self._settings = settings
        self._sleep = sleep

    @property
    def settings(self) -> RerankSettings:
        return self._settings

    def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> list[RerankResult]:
        """Return documents ordered by relevance, most relevant first."""
        if not documents:
            return []
        if not self._settings.api_key:
            raise ConfigurationIssue(
                "rerank",
                "no API key configured; set rerank.api_key in config.yaml "
                "or RAG__RERANK__API_KEY",
            )

        payload: dict = {
            "model": self._settings.model,
            "input": {"query": query, "documents": list(documents)},
        }
        if top_n is not None:
            payload["parameters"] = {"top_n": top_n}

        return self._request_with_retry(payload, len(documents))

    def _request_with_retry(self, payload: dict, expected: int) -> list[RerankResult]:
        attempts = max(1, self._settings.max_retries)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._request_once(payload, expected)
            except TransientFailure as exc:
                last = exc
                if attempt == attempts - 1:
                    break
                delay = 1.0 * (2**attempt)
                self._sleep(delay)
        assert last is not None
        raise last

    def _request_once(self, payload: dict, expected: int) -> list[RerankResult]:
        url = self._settings.url
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self._settings.timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise TransientFailure("rerank", "request timed out") from exc
        except httpx.HTTPError as exc:
            raise Unavailable("rerank", f"cannot reach {url}: {exc}") from exc

        if response.status_code != 200:
            self._raise_for_status(response)

        return self._parse(response, expected)

    def _raise_for_status(self, response: httpx.Response) -> None:
        detail = self._extract_message(response)
        status = response.status_code

        if status in _RETRYABLE_STATUS:
            raise TransientFailure("rerank", f"HTTP {status}: {detail}", status=status)
        if status in {401, 403}:
            raise ConfigurationIssue(
                "rerank",
                f"credentials rejected (HTTP {status}): {detail}. Check rerank.api_key.",
            )
        if status == 404:
            raise ConfigurationIssue(
                "rerank",
                f"endpoint not found (HTTP 404) at {self._settings.url}. "
                "Bailian reranking is only available on the native path "
                "/api/v1/services/rerank/text-rerank/text-rerank, never on "
                "/compatible-mode/v1/rerank.",
            )
        if status == 400:
            # On this endpoint a 400 usually means an invalid model, an empty
            # document list, or documents over the length limit.
            raise ConfigurationIssue(
                "rerank",
                f"request rejected (HTTP 400): {detail}. "
                f"The configured model is '{self._settings.model}'; it may not exist.",
            )
        raise ContractMismatch("rerank", f"unexpected HTTP {status}: {detail}")

    @staticmethod
    def _extract_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:300] or "(empty body)"
        if isinstance(body, dict):
            # Native DashScope shape.
            if body.get("message"):
                code = body.get("code")
                return f"{code}: {body['message']}" if code else str(body["message"])
            # Tolerate an OpenAI-shaped error too, in case a gateway wraps it.
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
        return str(body)[:300]

    def _parse(self, response: httpx.Response, expected: int) -> list[RerankResult]:
        try:
            body = response.json()
        except (ValueError) as exc:
            raise ContractMismatch("rerank", "response was not JSON") from exc

        if not isinstance(body, dict):
            raise ContractMismatch("rerank", "response is not a JSON object")

        output = body.get("output")
        if not isinstance(output, dict):
            raise ContractMismatch(
                "rerank",
                "response has no 'output' object; the endpoint may not be a "
                "text-rerank endpoint",
            )

        results = output.get("results")
        if not isinstance(results, list) or not results:
            raise ContractMismatch("rerank", "response has no 'output.results' array")

        parsed: list[RerankResult] = []
        for item in results:
            if not isinstance(item, dict):
                raise ContractMismatch("rerank", "result item is not an object")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int):
                raise ContractMismatch("rerank", f"result has a non-integer 'index': {index!r}")
            if not isinstance(score, (int, float)):
                raise ContractMismatch(
                    "rerank", f"result has a non-numeric 'relevance_score': {score!r}"
                )
            if not 0 <= index < expected:
                raise ContractMismatch(
                    "rerank", f"result index {index} is outside the {expected} submitted documents"
                )
            parsed.append(RerankResult(index=index, score=float(score)))

        if len({r.index for r in parsed}) != len(parsed):
            raise ContractMismatch("rerank", "response contains duplicate result indices")

        parsed.sort(key=lambda r: r.score, reverse=True)
        return parsed

    def verify(self) -> list[RerankResult]:
        """Exercise the endpoint with a case whose answer is unambiguous.

        Used by the startup self-check: responding at all is not enough, the
        ordering has to be sensible, or the service is not usable for reranking.
        """
        query = "猫"
        documents = ["猫是一种小型猫科动物。", "汽车需要汽油才能行驶。", "天空是蓝色的。"]
        results = self.rerank(query, documents)
        if not results:
            raise ContractMismatch("rerank", "self-check received no results")
        if results[0].index != 0:
            raise ContractMismatch(
                "rerank",
                "self-check found the ranking implausible: expected the passage "
                f"about cats to rank first, got index {results[0].index}. "
                f"Scores: {[(r.index, r.score) for r in results]}",
            )
        return results