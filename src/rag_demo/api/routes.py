"""HTTP API.

Two streaming endpoints and a health endpoint:

* ``POST /api/chat`` streams ``sources`` then ``token`` events. Sources go first on
  purpose: rewriting plus retrieval plus first-token latency can reach several
  seconds, and showing which documents are being consulted makes that wait legible.
* ``POST /api/docs`` streams ingestion progress, because a 30-page PDF takes tens of
  seconds and a silent request is indistinguishable from a hang.
* ``GET /api/health`` exposes per-dependency status so a configuration problem is
  identifiable without reading logs.

No provider credential or upstream endpoint is ever included in a response body.
"""

from __future__ import annotations

import json
import logging
import queue as queue_module
import threading
import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from rag_demo.api.services import Services, iter_sse
from rag_demo.errors import ProviderError
from rag_demo.generation.answer import NO_CONTEXT_ANSWER, describe_generation_error
from rag_demo.ingestion import IngestionError, Stage
from rag_demo.parsers.base import supported_extensions_display
from rag_demo.retrieval.hybrid import Context

logger = logging.getLogger(__name__)
router = APIRouter()


def _services(request: Request) -> Services:
    return request.app.state.services


def _sse(event: str, payload: dict[str, Any]) -> str:
    return next(iter_sse(event, json.dumps(payload, ensure_ascii=False)))


def _context_payload(context: Context, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "source": context.source_name,
        "page": context.page,
        "breadcrumb": context.breadcrumb,
        "matched_text": context.matched_text,
        "score": round(context.rerank_score, 4) if context.rerank_score is not None else None,
    }


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    """Report dependency status and knowledge-base size."""
    services = _services(request)
    report = services.selfcheck
    payload: dict[str, Any] = {
        "healthy": report.healthy if report else None,
        "dimension": services.dimension,
        "embedding_model": services.settings.embedding.model,
        "documents": 0,
        "units": services.lexical.size,
        "dependencies": [d.as_dict() for d in report.dependencies] if report else [],
    }
    try:
        payload["documents"] = len(services.registry.list_usable())
        payload["documents_failed"] = len(
            [d for d in services.registry.list_documents() if not d.usable]
        )
    except Exception as error:  # pragma: no cover - defensive
        payload["registry_error"] = str(error)
    return payload


@router.get("/api/docs")
def list_documents(request: Request) -> dict[str, Any]:
    services = _services(request)
    return {
        "documents": [record.as_dict() for record in services.registry.list_documents()]
    }


@router.delete("/api/docs/{doc_id}")
def delete_document(doc_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    from rag_demo.ingestion import remove_document

    try:
        removed = remove_document(doc_id, services.store, services.registry)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"could not remove document: {error}") from error

    if not removed:
        raise HTTPException(status_code=404, detail="no such document")

    # The keyword index still references the removed units; rebuilding keeps the two
    # retrieval channels describing the same corpus. The deletion has already
    # succeeded, so a rebuild failure must not turn it into a 500.
    try:
        services.refresh_lexical_index()
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("lexical index refresh failed after delete: %s", error)
    return {"removed": doc_id}


@router.post("/api/docs")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    """Ingest uploaded documents, streaming progress per file."""
    services = _services(request)

    accepted: list[tuple[str, bytes]] = []
    rejected: list[dict[str, str]] = []
    limit = services.settings.storage.max_upload_mb * 1024 * 1024

    for upload in files:
        name = upload.filename or "unnamed"
        content = await upload.read()
        if len(content) > limit:
            rejected.append(
                {
                    "name": name,
                    "error": f"file exceeds the {services.settings.storage.max_upload_mb} MB limit",
                }
            )
            continue
        accepted.append((name, content))

    def stream():
        for item in rejected:
            yield _sse("rejected", item)

        for name, content in accepted:
            # The id is decided here so the client learns it before ingestion
            # finishes and can correlate later updates.
            job_id = uuid.uuid4().hex[:12]
            yield _sse("start", {"name": name, "job": job_id})

            # Ingestion runs on a worker thread so progress can be streamed as it
            # happens. Draining the queue only after ingest() returned would make
            # every progress event arrive at the same moment, which is the same as
            # having no progress at all.
            events: queue_module.Queue = queue_module.Queue()
            outcome: dict[str, Any] = {}

            def on_progress(progress, job=job_id, filename=name):
                payload = progress.as_dict()
                payload.update({"name": filename, "job": job})
                events.put(("progress", payload))

            def worker(filename=name, payload=content):
                try:
                    outcome["result"] = services.pipeline.ingest(
                        filename, payload, on_progress=on_progress
                    )
                except IngestionError as error:
                    outcome["error"] = {
                        "stage": error.stage.value,
                        "error": error.message,
                    }
                except Exception as error:
                    outcome["error"] = {
                        "stage": Stage.PERSISTING.value,
                        "error": str(error),
                    }
                finally:
                    events.put(("finished", None))

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            while True:
                kind, payload = events.get()
                if kind == "finished":
                    break
                yield _sse(kind, payload)

            if "error" in outcome:
                yield _sse(
                    "failed",
                    {"name": name, "job": job_id, **outcome["error"]},
                )
                continue

            result = outcome["result"]
            try:
                services.refresh_lexical_index()
            except Exception as error:  # pragma: no cover - defensive
                logger.warning("lexical index refresh failed: %s", error)

            yield _sse(
                "done",
                {
                    "name": name,
                    "job": job_id,
                    "doc_id": result.doc_id,
                    "parents": result.parent_count,
                    "children": result.child_count,
                    "duration": round(result.duration_seconds, 2),
                },
            )

        yield _sse("complete", {"accepted": len(accepted), "rejected": len(rejected)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat")
async def chat(request: Request):
    """Answer a question, streaming sources then answer tokens."""
    services = _services(request)
    body = await request.json()

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    history = body.get("history") or []
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history must be a list")
    debug = bool(body.get("debug"))

    conversation = _to_conversation(services, history)

    def stream():
        yield _sse("status", {"stage": "rewriting"})

        query, rewritten = services.answerer.rewrite(question, conversation)
        if rewritten:
            yield _sse("rewritten", {"query": query})

        yield _sse("status", {"stage": "retrieving"})
        try:
            outcome = services.retriever.retrieve(query)
        except ProviderError as error:
            yield _sse("error", {"message": describe_retrieval_error(error)})
            return
        except Exception as error:
            logger.exception("retrieval failed")
            yield _sse("error", {"message": f"Retrieval failed: {error}"})
            return

        # Sources before tokens: this is the whole point of streaming them first.
        yield _sse(
            "sources",
            {
                "sources": [
                    _context_payload(context, index)
                    for index, context in enumerate(outcome.contexts, start=1)
                ],
                "rerank_applied": outcome.rerank_applied,
                "rerank_error": outcome.rerank_error,
            },
        )

        if debug:
            yield _sse("debug", _diagnostics(outcome))

        if not outcome.contexts:
            yield _sse("token", {"text": NO_CONTEXT_ANSWER})
            yield _sse("done", {"answer": NO_CONTEXT_ANSWER, "empty": True})
            return

        collected: list[str] = []
        try:
            for piece in services.answerer.stream_answer(question, outcome.contexts, conversation):
                collected.append(piece)
                yield _sse("token", {"text": piece})
        except Exception as error:
            # Say the answer is truncated rather than ending as though it were whole.
            message = describe_generation_error(error)
            yield _sse("error", {"message": message, "truncated": True})
            yield _sse("done", {"answer": "".join(collected), "truncated": True})
            return

        answer = "".join(collected)
        yield _sse("done", {"answer": answer, "truncated": False})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _to_conversation(services: Services, history: list) -> Any:
    """Rebuild conversation state from what the client sent.

    The server keeps no session state: the client owns the transcript, which keeps
    the API stateless and makes reloading the page harmless.
    """
    from rag_demo.generation.answer import Conversation

    conversation = Conversation(max_turns=services.settings.llm.history_turns)
    for item in history:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            conversation.append(question, answer)
    return conversation


def describe_retrieval_error(error: Exception) -> str:
    """Turn a retrieval failure into something a user can act on.

    Kept separate from the generation wording because telling someone their
    "generation is misconfigured" when the embedding provider is the problem sends
    them to the wrong place.
    """
    from rag_demo.errors import ConfigurationIssue, ContractMismatch, Unavailable

    if isinstance(error, ConfigurationIssue):
        return f"Retrieval is misconfigured: {error.message}"
    if isinstance(error, Unavailable):
        return f"A retrieval dependency is unreachable: {error.message}"
    if isinstance(error, ContractMismatch):
        return f"A retrieval dependency returned an unexpected response: {error.message}"
    return f"Retrieval failed: {type(error).__name__}: {error}"


def _diagnostics(outcome) -> dict[str, Any]:
    """Per-stage retrieval detail, for tuning rather than for users."""

    def summarise(candidates, limit=10):
        return [
            {
                "id": candidate.id,
                "parent_id": candidate.parent_id,
                "channels": candidate.channels,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "fused_rank": candidate.fused_rank,
                "fused_score": round(candidate.fused_score, 6),
                "rerank_rank": candidate.rerank_rank,
                "rerank_score": round(candidate.rerank_score, 6)
                if candidate.rerank_score is not None
                else None,
                "preview": candidate.text[:120],
            }
            for candidate in candidates[:limit]
        ]

    return {
        "query": outcome.query,
        "dense": summarise(outcome.dense),
        "sparse": summarise(outcome.sparse),
        "fused": summarise(outcome.fused),
        "reranked": summarise(outcome.reranked),
    }