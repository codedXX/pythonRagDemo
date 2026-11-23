"""Embedding provider adapter (Alibaba Cloud Bailian, OpenAI-compatible mode).

This endpoint speaks the OpenAI protocol, so it can be addressed with the
``openai`` SDK. Note the deliberate contrast with :mod:`rag_demo.embedding.rerank`,
which cannot: Bailian's reranking lives only on the native DashScope path and
``/compatible-mode/v1/rerank`` returns 404 (verified against the live service).

Batching, backoff and caching live here in :class:`Embedder`, while raw transport
is isolated in a client object so tests can substitute it.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from rag_demo.config import EmbeddingSettings
from rag_demo.errors import (
    ConfigurationIssue,
    ContractMismatch,
    TransientFailure,
    Unavailable,
)

# HTTP statuses that justify a retry. 400/401/403/404 mean the request itself is
# wrong and retrying only wastes the user's time and quota.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class EmbeddingClient:
    """Raw transport to an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, settings: EmbeddingSettings, timeout: float = 60.0) -> None:
        self._settings = settings
        self._timeout = timeout

    @property
    def settings(self) -> EmbeddingSettings:
        return self._settings

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in one request, returning vectors in input order."""
        if not texts:
            return []
        if not self._settings.api_key:
            raise ConfigurationIssue(
                "embedding",
                "no API key configured; set embedding.api_key in config.yaml "
                "or RAG__EMBEDDING__API_KEY",
            )

        url = self._settings.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self._settings.model, "input": list(texts)}
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise TransientFailure("embedding", f"request timed out after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise Unavailable("embedding", f"cannot reach {url}: {exc}") from exc

        if response.status_code != 200:
            self._raise_for_status(response)

        return self._parse(response, expected=len(texts))

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate an HTTP error into the right exception class."""
        detail = self._extract_message(response)
        status = response.status_code

        if status in _RETRYABLE_STATUS:
            raise TransientFailure("embedding", f"HTTP {status}: {detail}", status=status)

        if status in {401, 403}:
            raise ConfigurationIssue(
                "embedding",
                f"credentials rejected (HTTP {status}): {detail}. "
                "Check embedding.api_key.",
            )
        if status == 404:
            raise ConfigurationIssue(
                "embedding",
                f"endpoint or model not found (HTTP 404): {detail}. "
                f"Check embedding.base_url and embedding.model "
                f"(currently '{self._settings.model}').",
            )
        if status == 400 and _mentions_model(detail):
            raise ConfigurationIssue(
                "embedding",
                f"model rejected by the provider (HTTP 400): {detail}. "
                f"The configured model is '{self._settings.model}'; it may not exist "
                "or may not be available to this account.",
            )
        raise ContractMismatch("embedding", f"unexpected HTTP {status}: {detail}")

    @staticmethod
    def _extract_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return response.text[:300] or "(empty body)"
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if body.get("message"):
                return str(body["message"])
        return json.dumps(body)[:300]

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractMismatch("embedding", "response was not JSON") from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            raise ContractMismatch(
                "embedding", "response has no 'data' array of embeddings"
            )

        # The provider may reorder results; 'index' is authoritative.
        ordered: list[tuple[int, list[float]]] = []
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                raise ContractMismatch("embedding", "response item is not an object")
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ContractMismatch("embedding", "response item has no 'embedding' vector")
            if not all(isinstance(x, (int, float)) for x in vector):
                raise ContractMismatch("embedding", "embedding vector contains non-numeric values")
            index = item.get("index", position)
            if not isinstance(index, int):
                raise ContractMismatch("embedding", "response item has a non-integer 'index'")
            ordered.append((index, [float(x) for x in vector]))

        ordered.sort(key=lambda pair: pair[0])
        vectors = [vector for _, vector in ordered]

        if len(vectors) != expected:
            # A length mismatch would silently misalign vectors with chunks,
            # which is far worse than failing loudly.
            raise ContractMismatch(
                "embedding",
                f"asked for {expected} vectors but received {len(vectors)}",
            )

        dimensions = {len(v) for v in vectors}
        if len(dimensions) != 1:
            raise ContractMismatch(
                "embedding", f"inconsistent vector dimensions in one response: {sorted(dimensions)}"
            )

        return vectors


def _mentions_model(detail: str) -> bool:
    low = detail.lower()
    return any(
        token in low
        for token in ("model", "does not exist", "not found", "unsupported", "invalid_model")
    )


def content_hash(text: str, model: str) -> str:
    """Stable cache key for (text, model).

    Including the model means switching embedding models automatically misses the
    cache instead of silently reusing vectors from a different vector space.
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache:
    """Content-addressed on-disk cache of embedding vectors.

    Chunking parameters get tuned repeatedly, and every run re-embeds the same
    sentences otherwise. Caching by (text, model) makes tuning cost roughly only
    the text that actually changed.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by first two characters to keep directory sizes reasonable.
        return self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> list[float] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt cache entry is a cache miss, never a hard failure.
            return None
        vector = payload.get("vector")
        return vector if isinstance(vector, list) else None

    def put(self, key: str, vector: list[float], text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so an interrupted run cannot leave a truncated entry.
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps({"vector": vector, "text": text[:200]}, ensure_ascii=False)
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def stats(self) -> dict[str, int]:
        count = sum(1 for _ in self._dir.rglob("*.json"))
        return {"entries": count}


class Embedder:
    """Batching, retrying, caching facade over :class:`EmbeddingClient`."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        client: EmbeddingClient | None = None,
        cache: EmbeddingCache | None = None,
        sleep=time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client or EmbeddingClient(settings)
        self._cache = cache
        self._sleep = sleep

    @property
    def client(self) -> EmbeddingClient:
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``, using the cache and batching requests."""
        if not texts:
            return []

        model = self._settings.model
        keys = [content_hash(t, model) for t in texts]
        vectors: list[list[float] | None] = [None] * len(texts)

        if self._cache is not None:
            for position, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is not None:
                    vectors[position] = cached

        pending = [i for i, v in enumerate(vectors) if v is None]
        batch_size = max(1, self._settings.batch_size)
        for start in range(0, len(pending), batch_size):
            batch_indices = pending[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            produced = self._embed_with_retry(batch_texts)
            for index, vector in zip(batch_indices, produced, strict=True):
                vectors[index] = vector
                if self._cache is not None:
                    self._cache.put(keys[index], vector, texts[index])

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:  # pragma: no cover - defensive
            raise ContractMismatch("embedding", f"no vector produced for indices {missing}")

        return [v for v in vectors if v is not None]

    def _embed_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        attempts = max(1, self._settings.max_retries)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._client.embed(texts)
            except TransientFailure as exc:
                last = exc
                if attempt == attempts - 1:
                    break
                # Exponential backoff with jitter, so parallel callers do not
                # retry in lockstep.
                delay = self._settings.retry_base_delay_seconds * (2**attempt)
                self._sleep(delay + random.uniform(0, delay * 0.25))
            # ConfigurationIssue / ContractMismatch / Unavailable are not retried.
        assert last is not None
        raise last

    def probe_dimension(self, text: str = "ping") -> int:
        """Return the provider's vector dimension for the configured model.

        Called at startup because the dimension is fixed when a collection is
        created and cannot be changed afterwards. Probing turns "we guessed the
        dimension wrong" into a configuration error caught in the first minute.
        """
        vectors = self._client.embed([text])
        if not vectors:  # pragma: no cover - defensive
            raise ContractMismatch("embedding", "probe returned no vectors")
        return len(vectors[0])