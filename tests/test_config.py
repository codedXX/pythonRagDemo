"""Tests for configuration precedence: environment > config.yaml > default."""

from __future__ import annotations

import pytest

from rag_demo.config import (
    ConfigurationError,
    load_settings,
    missing_secrets,
    redacted,
)


def test_defaults_apply_when_no_file_and_no_env(tmp_path, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("RAG__"):
            monkeypatch.delenv(name, raising=False)

    settings = load_settings(root=tmp_path, use_env=False)

    assert settings.llm.model == "claude-sonnet-5"
    assert settings.milvus.consistency_level == "Strong"
    assert settings.retrieval.rrf_k == 60


def test_yaml_overrides_default(tmp_path, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("RAG__"):
            monkeypatch.delenv(name, raising=False)

    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  rrf_k: 100\nllm:\n  model: from-yaml\n",
        encoding="utf-8",
    )

    settings = load_settings(root=tmp_path, use_env=False)

    assert settings.retrieval.rrf_k == 100
    assert settings.llm.model == "from-yaml"
    # Untouched keys still fall back to defaults.
    assert settings.retrieval.dense_top_k == 40


def test_env_overrides_yaml(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "llm:\n  api_key: from-yaml\n  model: from-yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG__LLM__API_KEY", "from-env")
    monkeypatch.setenv("RAG__RETRIEVAL__RRF_K", "77")

    settings = load_settings(root=tmp_path)

    assert settings.llm.api_key == "from-env"
    assert settings.retrieval.rrf_k == 77
    # Env did not touch the model, so YAML still wins over the default.
    assert settings.llm.model == "from-yaml"


def test_env_coerces_types(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG__CHUNKING__SEMANTIC_ENABLED", "false")
    monkeypatch.setenv("RAG__CHUNKING__PARENT_MAX_CHARS", "1500")
    monkeypatch.setenv("RAG__LLM__TEMPERATURE", "0.9")

    settings = load_settings(root=tmp_path)

    assert settings.chunking.semantic_enabled is False
    assert settings.chunking.parent_max_chars == 1500
    assert settings.llm.temperature == pytest.approx(0.9)


def test_env_ignored_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG__LLM__MODEL", "from-env")

    settings = load_settings(root=tmp_path, use_env=False)

    assert settings.llm.model == "claude-sonnet-5"


def test_invalid_yaml_reports_configuration_error(tmp_path):
    (tmp_path / "config.yaml").write_text("llm: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_settings(root=tmp_path, use_env=False)


def test_invalid_types_reports_configuration_error(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "retrieval:\n  rrf_k: not-a-number\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError):
        load_settings(root=tmp_path, use_env=False)


def test_redacted_hides_secrets(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "llm:\n  api_key: sk-super-secret\nembedding:\n  api_key: sk-other\n",
        encoding="utf-8",
    )
    settings = load_settings(root=tmp_path, use_env=False)

    view = redacted(settings)
    blob = repr(view)

    assert "sk-super-secret" not in blob
    assert "sk-other" not in blob
    assert view["llm"]["api_key"] == "***set***"


def test_missing_secrets_names_only_absent_keys(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "llm:\n  api_key: present\n", encoding="utf-8"
    )
    settings = load_settings(root=tmp_path, use_env=False)

    absent = missing_secrets(settings)

    assert "llm" not in absent
    assert absent["embedding"] == ["api_key"]
    assert absent["rerank"] == ["api_key"]
    # Milvus token is optional for a local server.
    assert "milvus" not in absent