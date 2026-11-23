"""Application wiring: one place where settings become services.

Constructed once per process. The pieces are plain objects rather than globals so
tests can build a fully isolated instance with fake providers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag_demo.chunking.chunker import Chunker
from rag_demo.config import Settings
from rag_demo.embedding.provider import Embedder
from rag_demo.embedding.rerank import Reranker
from rag_demo.generation.answer import Answerer
from rag_demo.generation.llm import Generator
from rag_demo.ingestion import IngestionPipeline
from rag_demo.retrieval.hybrid import HybridRetriever
from rag_demo.retrieval.lexical import LexicalIndex
from rag_demo.selfcheck import SelfCheckReport, run_selfcheck
from rag_demo.store.milvus_store import MilvusStore
from rag_demo.store.registry import DocumentRegistry

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Everything the API needs, assembled."""

    settings: Settings
    store: MilvusStore
    registry: DocumentRegistry
    embedder: Embedder
    reranker: Reranker
    generator: Generator
    lexical: LexicalIndex
    retriever: HybridRetriever
    answerer: Answerer
    pipeline: IngestionPipeline
    selfcheck: SelfCheckReport | None = None
    dimension: int | None = None
    _lexical_lock: threading.Lock = field(default_factory=threading.Lock)

    def refresh_lexical_index(self) -> int:
        """Rebuild the keyword index from the store.

        Called after ingestion and removal. The lock keeps two concurrent uploads
        from rebuilding over each other and leaving a half-populated index.
        """
        with self._lexical_lock:
            return self.retriever.rebuild_lexical_index()

    def ensure_collections(self) -> None:
        """Create collections if needed, discovering the dimension first."""
        if self.dimension is None:
            self.dimension = self.embedder.probe_dimension("ping")
        self.store.ensure_collections(dimension=self.dimension)
        self.store.write_meta(
            {
                "embedding_model": self.settings.embedding.model,
                "dimension": self.dimension,
            }
        )


def build_services(settings: Settings) -> Services:
    """Assemble the service graph.

    The dimension is discovered from the provider rather than configured, because it
    is fixed at collection-creation time and a wrong guess is only discovered much
    later.
    """
    store = MilvusStore(settings)
    registry = DocumentRegistry(settings.sqlite_path())
    embedder = Embedder(settings.embedding)
    reranker = Reranker(settings.rerank)
    generator = Generator(settings.llm)
    lexical = LexicalIndex()

    retriever = HybridRetriever(settings.retrieval, store, embedder, lexical, reranker)
    answerer = Answerer(settings, generator)
    pipeline = IngestionPipeline(
        settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)
    )

    return Services(
        settings=settings,
        store=store,
        registry=registry,
        embedder=embedder,
        reranker=reranker,
        generator=generator,
        lexical=lexical,
        retriever=retriever,
        answerer=answerer,
        pipeline=pipeline,
    )


def initialise(services: Services, verify: bool = True) -> SelfCheckReport | None:
    """Prepare the process for serving traffic.

    Failures here are logged rather than raised: the service must still start when a
    provider is misconfigured so the interface can explain what is wrong, instead of
    the process dying and leaving nothing to inspect.
    """
    report: SelfCheckReport | None = None
    if verify:
        try:
            report = run_selfcheck(services.settings, include_llm=False)
            services.selfcheck = report
            for dependency in report.dependencies:
                logger.info(
                    "dependency %s: %s (%s)",
                    dependency.name,
                    dependency.status.value,
                    dependency.message,
                )
        except Exception as error:  # pragma: no cover - never fatal
            logger.warning("self-check failed to run: %s", error)

    dimension = services.selfcheck.embedding_dimension if services.selfcheck else None
    if dimension is None:
        try:
            dimension = services.embedder.probe_dimension("ping")
        except Exception as error:
            logger.warning(
                "cannot discover embedding dimension (%s); collections are not "
                "initialised until a provider is reachable",
                error,
            )
            return report

    try:
        services.dimension = dimension
        services.store.ensure_collections(dimension=dimension)
        services.store.write_meta(
            {"embedding_model": services.settings.embedding.model, "dimension": dimension}
        )
        count = services.refresh_lexical_index()
        logger.info("lexical index rebuilt with %s units", count)
    except Exception as error:  # pragma: no cover - never fatal at startup
        logger.warning("store initialisation failed: %s", error)

    return report


def iter_sse(event: str, data: str) -> Iterator[str]:
    """Format a Server-Sent Event frame.

    Multi-line payloads are sent as repeated ``data:`` lines, which is what the
    browser's EventSource concatenates back into one string.
    """
    lines = [f"event: {event}"]
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    return iter(["\n".join(lines) + "\n\n"])