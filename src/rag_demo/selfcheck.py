"""Startup verification of external dependencies.

The purpose is to convert "we think these model names exist" into a definite,
actionable answer before any document is ingested. Two unknowns motivated this:

* The configured embedding and reranking model names could not be verified
  against the live service while the change was designed.
* The embedding dimension is fixed when a collection is created, so guessing it
  wrong means rebuilding everything.

The self-check never raises for an unhealthy dependency: it reports it, so the
service still starts and the interface can explain what is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rag_demo.config import Settings, missing_secrets
from rag_demo.embedding.provider import Embedder
from rag_demo.embedding.rerank import Reranker
from rag_demo.errors import (
    ConfigurationIssue,
    ContractMismatch,
    ProviderError,
    TransientFailure,
    Unavailable,
)
from rag_demo.generation.llm import Generator


class Status(str, Enum):
    HEALTHY = "healthy"
    MISCONFIGURED = "misconfigured"
    UNREACHABLE = "unreachable"
    CONTRACT_MISMATCH = "contract_mismatch"


@dataclass(slots=True)
class DependencyReport:
    """Outcome of verifying one dependency."""

    name: str
    status: Status
    message: str
    detail: dict = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status is Status.HEALTHY

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(slots=True)
class SelfCheckReport:
    """Outcome of verifying every dependency."""

    dependencies: list[DependencyReport]
    embedding_dimension: int | None = None
    embedding_model: str = ""
    stored: dict | None = None

    @property
    def healthy(self) -> bool:
        return all(d.healthy for d in self.dependencies)

    def by_name(self, name: str) -> DependencyReport | None:
        return next((d for d in self.dependencies if d.name == name), None)

    def as_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "dependencies": [d.as_dict() for d in self.dependencies],
            "stored": self.stored,
        }


def _status_for(exc: Exception) -> Status:
    if isinstance(exc, ConfigurationIssue):
        return Status.MISCONFIGURED
    if isinstance(exc, Unavailable):
        return Status.UNREACHABLE
    if isinstance(exc, ContractMismatch):
        return Status.CONTRACT_MISMATCH
    if isinstance(exc, TransientFailure):
        return Status.UNREACHABLE
    return Status.UNREACHABLE


def check_embedding(settings: Settings) -> DependencyReport:
    """Probe the embedding provider and discover the vector dimension."""
    embedder = Embedder(settings.embedding)
    try:
        dimension = embedder.probe_dimension("ping")
    except ProviderError as exc:
        return DependencyReport(
            name="embedding",
            status=_status_for(exc),
            message=str(exc),
            detail={"model": settings.embedding.model},
        )
    except Exception as exc:  # pragma: no cover - unexpected transport shapes
        return DependencyReport(
            name="embedding",
            status=Status.UNREACHABLE,
            message=f"unexpected failure: {exc}",
            detail={"model": settings.embedding.model},
        )

    return DependencyReport(
        name="embedding",
        status=Status.HEALTHY,
        message=f"model available, dimension {dimension}",
        detail={"model": settings.embedding.model, "dimension": dimension},
    )


def check_rerank(settings: Settings) -> DependencyReport:
    """Verify the reranker answers and ranks a unambiguous case sensibly."""
    try:
        results = Reranker(settings.rerank).verify()
    except ProviderError as exc:
        return DependencyReport(
            name="rerank",
            status=_status_for(exc),
            message=str(exc),
            detail={"model": settings.rerank.model},
        )
    except Exception as exc:  # pragma: no cover
        return DependencyReport(
            name="rerank",
            status=Status.UNREACHABLE,
            message=f"unexpected failure: {exc}",
            detail={"model": settings.rerank.model},
        )

    return DependencyReport(
        name="rerank",
        status=Status.HEALTHY,
        message="model available, ranking plausible",
        detail={
            "model": settings.rerank.model,
            "top_index": results[0].index,
            "top_score": round(results[0].score, 4),
        },
    )


def check_llm(settings: Settings) -> DependencyReport:
    """Verify the generation model exists, is authorised and streams."""
    try:
        text = Generator(settings.llm).verify()
    except ProviderError as exc:
        return DependencyReport(
            name="llm",
            status=_status_for(exc),
            message=str(exc),
            detail={"model": settings.llm.model, "base_url": settings.llm.base_url},
        )
    except Exception as exc:  # pragma: no cover
        return DependencyReport(
            name="llm",
            status=Status.UNREACHABLE,
            message=f"unexpected failure: {exc}",
            detail={"model": settings.llm.model},
        )

    return DependencyReport(
        name="llm",
        status=Status.HEALTHY,
        message="model available, streaming confirmed",
        detail={"model": settings.llm.model, "sample": text[:40]},
    )


def check_milvus(settings: Settings) -> DependencyReport:
    """Verify the vector store is reachable and its version is usable."""
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus.uri, token=settings.milvus.token or None)
        version = client.get_server_version()
    except ImportError as exc:  # pragma: no cover
        return DependencyReport(
            name="milvus",
            status=Status.MISCONFIGURED,
            message=f"pymilvus is not installed: {exc}",
        )
    except Exception as exc:
        return DependencyReport(
            name="milvus",
            status=Status.UNREACHABLE,
            message=f"cannot reach {settings.milvus.uri}: {exc}",
            detail={"uri": settings.milvus.uri},
        )

    return DependencyReport(
        name="milvus",
        status=Status.HEALTHY,
        message=f"server {version} reachable",
        detail={"uri": settings.milvus.uri, "server_version": version},
    )


def check_stored_dimension(settings: Settings, dimension: int | None) -> DependencyReport | None:
    """Compare the probed dimension with what existing collections were built for.

    Returns ``None`` when there is no stored metadata yet, which is the normal
    first-run case.
    """
    from rag_demo.store.milvus_store import MilvusStore

    if dimension is None:
        return None

    store = MilvusStore(settings)
    try:
        stored = store.read_meta()
    except Exception as exc:
        return DependencyReport(
            name="store_schema",
            status=Status.UNREACHABLE,
            message=f"cannot read stored metadata: {exc}",
        )

    if not stored:
        return None

    stored_model = stored.get("embedding_model")
    stored_dim = stored.get("dimension")
    if stored_model == settings.embedding.model and stored_dim == dimension:
        return DependencyReport(
            name="store_schema",
            status=Status.HEALTHY,
            message=f"stored schema matches configuration ({stored_model}, dim {stored_dim})",
            detail={"embedding_model": stored_model, "dimension": stored_dim},
        )

    return DependencyReport(
        name="store_schema",
        status=Status.MISCONFIGURED,
        message=(
            "existing storage was built with a different embedding configuration: "
            f"stored model '{stored_model}' (dim {stored_dim}) but the provider now "
            f"reports '{settings.embedding.model}' (dim {dimension}). "
            "Vectors of different models or dimensions cannot be mixed; rebuild the "
            "knowledge base (drop the rag_child / rag_parent / rag_meta collections) "
            "before ingesting."
        ),
        detail={
            "stored_model": stored_model,
            "stored_dimension": stored_dim,
            "current_model": settings.embedding.model,
            "current_dimension": dimension,
        },
    )


def run_selfcheck(settings: Settings, include_llm: bool = True) -> SelfCheckReport:
    """Verify every dependency and return a consolidated report."""
    absent = missing_secrets(settings)

    embedding = check_embedding(settings)
    dimension = embedding.detail.get("dimension") if embedding.healthy else None

    rerank = check_rerank(settings)
    milvus = check_milvus(settings)
    dependencies = [embedding, rerank, milvus]

    if include_llm:
        dependencies.append(check_llm(settings))

    # Surface missing keys as their own note rather than leaving the user to
    # infer it from three separate provider rejections.
    for section, keys in absent.items():
        dependencies.append(
            DependencyReport(
                name=f"{section}_credentials",
                status=Status.MISCONFIGURED,
                message=(
                    f"{section} has no {', '.join(keys)} configured; "
                    f"fill it in config.yaml or set RAG__{section.upper()}__API_KEY"
                ),
            )
        )

    stored_report = None
    try:
        stored_report = check_stored_dimension(settings, dimension)
    except Exception:  # pragma: no cover - store may not exist yet
        stored_report = None
    if stored_report is not None:
        dependencies.append(stored_report)

    report = SelfCheckReport(
        dependencies=dependencies,
        embedding_dimension=dimension,
        embedding_model=settings.embedding.model,
    )
    return report


def format_report(report: SelfCheckReport) -> str:
    """Render a report for the terminal."""
    lines = []
    for dependency in report.dependencies:
        marker = {
            Status.HEALTHY: "OK  ",
            Status.MISCONFIGURED: "FAIL",
            Status.UNREACHABLE: "DOWN",
            Status.CONTRACT_MISMATCH: "BAD ",
        }[dependency.status]
        lines.append(f"[{marker}] {dependency.name:<22} {dependency.message}")

    lines.append("")
    if report.embedding_dimension is not None:
        lines.append(
            f"embedding: {report.embedding_model} -> dimension "
            f"{report.embedding_dimension}"
        )
    lines.append("result: " + ("all dependencies healthy" if report.healthy else "problems found"))
    return "\n".join(lines)


def write_meta(settings: Settings, dimension: int) -> None:
    """Persist the embedding model and dimension used to build storage."""
    from rag_demo.store.milvus_store import MilvusStore

    MilvusStore(settings).write_meta(
        {
            "embedding_model": settings.embedding.model,
            "dimension": dimension,
        }
    )


def report_to_json(report: SelfCheckReport) -> str:
    return json.dumps(report.as_dict(), ensure_ascii=False, indent=2)


def default_config_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / "config.yaml"