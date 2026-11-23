"""Configuration loading with a three-level precedence chain.

Precedence, highest first:

1. Environment variables named ``RAG__<SECTION>__<FIELD>`` (double underscores
   separate path segments, so ``RAG__EMBEDDING__API_KEY`` sets
   ``embedding.api_key``).
2. ``config.yaml`` next to the project root.
3. The defaults declared in the models below.

The reason environment variables win is that they make containerised or CI runs
possible without editing a secret-bearing file, while the hand-edited YAML file
stays the convenient local workflow.

Secrets are never logged or echoed by anything in this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ENV_PREFIX = "RAG__"

# Keys that must never appear in logs, error messages or health output.
SECRET_FIELDS = frozenset({"api_key", "token"})


class ConfigurationError(RuntimeError):
    """Raised when configuration is missing or internally inconsistent."""


class LLMSettings(BaseModel):
    base_url: str = "http://8.134.19.201:20089/v1"
    api_key: str = ""
    model: str = "claude-sonnet-5"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    history_turns: int = 6


class EmbeddingSettings(BaseModel):
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen3.7-text-embedding-flash"
    batch_size: int = 10
    max_retries: int = 5
    retry_base_delay_seconds: float = 1.0


class RerankSettings(BaseModel):
    base_url: str = "https://dashscope.aliyuncs.com"
    path: str = "/api/v1/services/rerank/text-rerank/text-rerank"
    api_key: str = ""
    model: str = "qwen3.7-text-rerank"
    timeout_seconds: float = 60.0
    max_retries: int = 3

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/" + self.path.lstrip("/")


class MilvusSettings(BaseModel):
    uri: str = "http://127.0.0.1:19530"
    token: str = ""
    child_collection: str = "rag_child"
    parent_collection: str = "rag_parent"
    meta_collection: str = "rag_meta"
    # Strong is mandatory; see the explanatory comment in config.example.yaml.
    consistency_level: str = "Strong"


class ChunkingSettings(BaseModel):
    parent_target_chars: int = 1200
    parent_max_chars: int = 2000
    child_target_chars: int = 320
    child_overlap_chars: int = 60
    semantic_enabled: bool = True
    semantic_break_percentile: int = 88
    semantic_window_chars: int = 180


class RetrievalSettings(BaseModel):
    dense_top_k: int = 40
    sparse_top_k: int = 40
    rrf_k: int = 60
    rerank_candidates: int = 30
    context_count: int = 6
    context_min_count: int = 3
    context_char_budget: int = 8000
    parent_excerpt_chars: int = 1800


class ParsingSettings(BaseModel):
    min_chars_per_page: int = 40


class StorageSettings(BaseModel):
    sqlite_path: str = "data/rag.db"
    embedding_cache_path: str = "data/embedding_cache"
    max_upload_mb: int = 50


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    # Populated by the loader; not user-configurable.
    root: Path = Field(default_factory=Path.cwd, exclude=True)

    def data_dir(self) -> Path:
        return self.root / "data"

    def sqlite_path(self) -> Path:
        return self.root / self.storage.sqlite_path

    def embedding_cache_dir(self) -> Path:
        return self.root / self.storage.embedding_cache_path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` recursively, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _coerce(raw: str, current: Any) -> Any:
    """Coerce an environment string to the type of the value it replaces."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply ``RAG__<SECTION>__<FIELD>`` environment variables over ``data``."""
    result = dict(data)
    for env_name, env_value in os.environ.items():
        if not env_name.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in env_name[len(ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue

        cursor: Any = result
        for part in parts[:-1]:
            node = cursor.get(part)
            if not isinstance(node, dict):
                node = {}
                cursor[part] = node
            cursor = node

        leaf = parts[-1]
        cursor[leaf] = _coerce(env_value, cursor.get(leaf))
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"{path.name} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{path.name} must contain a mapping at the top level")
    return loaded


def load_settings(
    root: Path | str | None = None,
    config_path: Path | str | None = None,
    use_env: bool = True,
) -> Settings:
    """Load settings for ``root``.

    ``config_path`` overrides the default ``<root>/config.yaml``. Passing
    ``use_env=False`` ignores the environment, which keeps tests deterministic.
    """
    resolved_root = Path(root) if root is not None else Path.cwd()
    resolved_config = (
        Path(config_path)
        if config_path is not None
        else resolved_root / "config.yaml"
    )

    data = _read_yaml(resolved_config)
    if use_env:
        data = _apply_env_overrides(data)
    data.setdefault("root", resolved_root)

    try:
        return Settings.model_validate(data)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


def redacted(settings: Settings) -> dict[str, Any]:
    """Return a loggable view of ``settings`` with secrets replaced.

    Used by health output and startup logging so a key can never leak into the
    browser or a log file.
    """
    view = settings.model_dump(exclude={"root"})
    for section, values in view.items():
        if not isinstance(values, dict):
            continue
        for key in list(values):
            if key in SECRET_FIELDS:
                values[key] = "***set***" if values[key] else ""
    return view


def missing_secrets(settings: Settings) -> dict[str, list[str]]:
    """Report which secret fields are unset, keyed by section.

    Deliberately reports only *which* keys are absent, never their values, so
    the result is safe to expose through the health endpoint.
    """
    sections = {
        "llm": settings.llm,
        "embedding": settings.embedding,
        "rerank": settings.rerank,
        "milvus": settings.milvus,
    }
    absent: dict[str, list[str]] = {}
    for name, section in sections.items():
        missing = [
            key
            for key in SECRET_FIELDS
            if hasattr(section, key) and not getattr(section, key)
            # The Milvus token is optional for a local unauthenticated server.
            and not (name == "milvus" and key == "token")
        ]
        if missing:
            absent[name] = missing
    return absent