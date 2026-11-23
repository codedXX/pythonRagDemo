"""Reciprocal Rank Fusion and the retrieval pipeline.

Fusion is rank-based:

    score(d) = sum over channels of 1 / (k + rank_channel(d))

Rank-based rather than score-based because the two channels produce incomparable
numbers: BM25 scores are unbounded and corpus-dependent, cosine similarities are
bounded, and their distributions shift with every document added. Normalising them
into a shared scale would introduce parameters that must themselves be tuned per
corpus. Ranks are always comparable.

The pipeline then reranks and expands:

    query ─┬─ dense  ─┐
           └─ sparse ─┴─ RRF ─→ rerank ─→ dedupe by parent ─→ expand ─→ contexts

Two subtleties that are easy to get wrong and are therefore explicit in the code:

* The reranker receives more candidates than generation needs. After collapsing
  siblings that share a parent, a reranked top-8 can easily become three distinct
  contexts, so the supply is backfilled from what remains.
* What is sent to the reranker is breadcrumb + text, not bare text. A fragment like
  "as the table below shows" is unrankable on its own, and scoring it that way
  unfairly penalises passages whose meaning lives in their heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_demo.config import RetrievalSettings
from rag_demo.embedding.provider import Embedder
from rag_demo.embedding.rerank import Reranker
from rag_demo.errors import ProviderError
from rag_demo.retrieval.lexical import LexicalIndex
from rag_demo.store.milvus_store import MilvusStore


@dataclass(slots=True)
class Candidate:
    """A retrieval candidate, carrying the provenance needed for citation."""

    id: str
    parent_id: str
    text: str
    breadcrumb: str
    page: int
    source_name: str
    doc_id: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0
    fused_rank: int = 0
    rerank_score: float | None = None
    rerank_rank: int | None = None

    @property
    def channels(self) -> list[str]:
        """Which channels found this candidate. Shown in diagnostics."""
        found = []
        if self.dense_rank is not None:
            found.append("dense")
        if self.sparse_rank is not None:
            found.append("sparse")
        return found

    def rerank_text(self) -> str:
        """The text submitted to the reranker: context header plus body."""
        if self.breadcrumb:
            return f"{self.source_name} > {self.breadcrumb} | {self.text}"
        return f"{self.source_name} | {self.text}"


@dataclass(slots=True)
class Context:
    """A parent unit selected for generation, with the child that matched."""

    parent_id: str
    text: str
    breadcrumb: str
    page: int
    source_name: str
    matched_child: str = ""
    matched_text: str = ""
    rerank_score: float | None = None


@dataclass(slots=True)
class RetrievalOutcome:
    """Everything a caller may want to know about one retrieval."""

    query: str
    contexts: list[Context] = field(default_factory=list)
    dense: list[Candidate] = field(default_factory=list)
    sparse: list[Candidate] = field(default_factory=list)
    fused: list[Candidate] = field(default_factory=list)
    reranked: list[Candidate] = field(default_factory=list)
    rerank_applied: bool = True
    rerank_error: str | None = None

    @property
    def empty(self) -> bool:
        return not self.contexts


def rrf_fuse(
    rankings: list[list[str]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one, highest score first.

    ``rankings`` holds one list per channel in descending relevance. An id absent
    from a channel contributes nothing for that channel, which is what makes a
    document found by both channels outrank one found by only one.
    """
    scores: dict[str, float] = {}
    for channel_index, ranking in enumerate(rankings):
        weight = weights[channel_index] if weights else 1.0
        # A channel contributes once per document, at its best rank within that
        # channel. Counting a repeated id twice would let a single channel inflate
        # its own candidate above one found by both channels.
        best_rank: dict[str, int] = {}
        for rank, identifier in enumerate(ranking):
            if identifier not in best_rank:
                best_rank[identifier] = rank
        for identifier, rank in best_rank.items():
            scores[identifier] = scores.get(identifier, 0.0) + weight / (rrf_k + rank + 1)

    # Ties broken by id so results are deterministic, which matters when comparing
    # retrieval configurations.
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """Dense + sparse recall, RRF fusion, reranking, parent expansion."""

    def __init__(
        self,
        settings: RetrievalSettings,
        store: MilvusStore,
        embedder: Embedder,
        lexical: LexicalIndex,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._lexical = lexical
        self._reranker = reranker

    # -------------------------------------------------------------------- recall

    def _dense(self, query: str) -> list[Candidate]:
        vectors = self._embedder.embed_texts([query])
        if not vectors:
            return []
        rows = self._store.search_children(vectors[0], limit=self._settings.dense_top_k)
        candidates: list[Candidate] = []
        for rank, row in enumerate(rows):
            candidates.append(
                Candidate(
                    id=row["id"],
                    parent_id=row["parent_id"],
                    text=row.get("text", ""),
                    breadcrumb=row.get("breadcrumb", ""),
                    page=int(row.get("page", 0) or 0),
                    source_name=row.get("source_name", ""),
                    doc_id=row.get("doc_id", ""),
                    dense_rank=rank,
                )
            )
        return candidates

    def _sparse(self, query: str) -> list[Candidate]:
        hits = self._lexical.search(query, limit=self._settings.sparse_top_k)
        if not hits:
            return []

        ids = [hit.id for hit in hits]
        rows = self._fetch_children(ids)
        candidates: list[Candidate] = []
        for rank, hit in enumerate(hits):
            row = rows.get(hit.id)
            if row is None:
                # The lexical index can briefly contain an id the store no longer
                # has; skipping is correct, surfacing nothing about it is honest.
                continue
            candidates.append(
                Candidate(
                    id=hit.id,
                    parent_id=row["parent_id"],
                    text=row.get("text", ""),
                    breadcrumb=row.get("breadcrumb", ""),
                    page=int(row.get("page", 0) or 0),
                    source_name=row.get("source_name", ""),
                    doc_id=row.get("doc_id", ""),
                    sparse_rank=rank,
                )
            )
        return candidates

    def _fetch_children(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        quoted = ", ".join(f'"{i}"' for i in ids)
        rows = self._store.client.query(
            self._store.child_name,
            filter=f"id in [{quoted}] and active == true",
            output_fields=[
                "id",
                "doc_id",
                "parent_id",
                "text",
                "breadcrumb",
                "page",
                "source_name",
            ],
            limit=max(1, len(ids)),
        )
        return {row["id"]: row for row in rows}

    # --------------------------------------------------------------------- fuse

    def _fuse(
        self, dense: list[Candidate], sparse: list[Candidate]
    ) -> list[Candidate]:
        by_id: dict[str, Candidate] = {}
        for candidate in dense:
            by_id[candidate.id] = candidate
        for candidate in sparse:
            if candidate.id in by_id:
                # Same unit found by both channels: record both ranks, which is
                # what makes it rank higher after fusion.
                by_id[candidate.id].sparse_rank = candidate.sparse_rank
            else:
                by_id[candidate.id] = candidate

        fused = rrf_fuse(
            [
                [c.id for c in dense],
                [c.id for c in sparse],
            ],
            rrf_k=self._settings.rrf_k,
        )

        result: list[Candidate] = []
        for rank, (identifier, score) in enumerate(fused):
            candidate = by_id.get(identifier)
            if candidate is None:  # pragma: no cover - defensive
                continue
            candidate.fused_score = score
            candidate.fused_rank = rank
            result.append(candidate)
        return result

    # ------------------------------------------------------------------- rerank

    def _rerank(self, query: str, fused: list[Candidate]) -> tuple[list[Candidate], bool, str | None]:
        """Rerank the top fused candidates.

        Returns ``(candidates, applied, error)``. A reranker failure degrades to the
        fused order rather than failing retrieval, because the fused result is still
        usable and losing the whole answer would be worse.
        """
        if self._reranker is None:
            return fused, False, "no reranker configured"

        subset = fused[: self._settings.rerank_candidates]
        if not subset:
            return fused, False, None

        try:
            results = self._reranker.rerank(query, [c.rerank_text() for c in subset])
        except ProviderError as error:
            return fused, False, str(error)
        except Exception as error:  # pragma: no cover - defensive
            return fused, False, f"{type(error).__name__}: {error}"

        reranked: list[Candidate] = []
        for rank, result in enumerate(results):
            if not 0 <= result.index < len(subset):
                continue
            candidate = subset[result.index]
            candidate.rerank_score = result.score
            candidate.rerank_rank = rank
            reranked.append(candidate)

        if not reranked:  # pragma: no cover - defensive
            return fused, False, "reranker returned no usable results"

        # Anything the reranker did not score keeps its fused order and follows.
        scored = {c.id for c in reranked}
        reranked.extend(c for c in fused if c.id not in scored)
        return reranked, True, None

    # ------------------------------------------------------------------- expand

    def _expand(self, candidates: list[Candidate]) -> list[Context]:
        """Turn selected children into distinct parent contexts.

        Several children usually share a parent. Including the parent once keeps the
        prompt from repeating the same passage and wasting the context budget, and
        the supply is backfilled so the configured number of contexts is met.
        """
        contexts: list[Context] = []
        seen: set[str] = set()
        pending: list[Candidate] = []

        for candidate in candidates:
            if candidate.parent_id in seen:
                continue
            seen.add(candidate.parent_id)
            pending.append(candidate)
            if len(pending) >= self._settings.context_count:
                break

        if not pending:
            return []

        parents = self._store.fetch_parents([c.parent_id for c in pending])

        for candidate in pending:
            parent = parents.get(candidate.parent_id)
            if parent is None:
                # The parent is missing, which would mean the store is inconsistent.
                # Falling back to the child keeps the result usable and honest.
                text = candidate.text
                breadcrumb = candidate.breadcrumb
                page = candidate.page
            else:
                text = parent.get("text", "")
                breadcrumb = parent.get("breadcrumb") or candidate.breadcrumb
                page = int(parent.get("page_start", 0) or candidate.page)
            contexts.append(
                Context(
                    parent_id=candidate.parent_id,
                    text=self._bound(text, candidate.text),
                    breadcrumb=breadcrumb,
                    page=page,
                    source_name=candidate.source_name,
                    matched_child=candidate.id,
                    matched_text=candidate.text,
                    rerank_score=candidate.rerank_score,
                )
            )

        # Enforce the total budget across contexts rather than per context alone.
        return self._apply_budget(contexts)

    def _bound(self, parent_text: str, child_text: str) -> str:
        """Reduce an oversized parent to an excerpt around the matched child.

        Passing a 3000-character parent verbatim would crowd out the other contexts,
        so the region around the match is kept instead.
        """
        limit = self._settings.parent_excerpt_chars
        if len(parent_text) <= limit:
            return parent_text

        position = parent_text.find(child_text) if child_text else -1
        if position < 0:
            return parent_text[:limit].rstrip() + "…"

        half = max(1, (limit - len(child_text)) // 2)
        start = max(0, position - half)
        end = min(len(parent_text), position + len(child_text) + half)
        excerpt = parent_text[start:end].strip()
        if start > 0:
            excerpt = "…" + excerpt
        if end < len(parent_text):
            excerpt = excerpt + "…"
        return excerpt

    def _apply_budget(self, contexts: list[Context]) -> list[Context]:
        """Trim contexts to the configured character budget, keeping the best."""
        budget = self._settings.context_char_budget
        kept: list[Context] = []
        used = 0
        for context in contexts:
            remaining = budget - used
            if remaining <= 200:
                break
            if len(context.text) > remaining:
                context.text = context.text[:remaining].rstrip() + "…"
            kept.append(context)
            used += len(context.text)
            if len(kept) >= self._settings.context_count:
                break
        return kept

    # --------------------------------------------------------------------- main

    def retrieve(self, query: str) -> RetrievalOutcome:
        """Run the full pipeline for one query."""
        outcome = RetrievalOutcome(query=query)

        # An empty keyword index usually means nothing has been ingested. Checking
        # here avoids an embedding call that cannot match anything, and means a
        # question asked before any upload gets an honest "no content" answer rather
        # than a configuration error.
        if self._lexical.size == 0:
            if self._store_stats_are_empty():
                return outcome
            # The store has data but the index does not: the index is stale (a
            # rebuild failed, say). Resynchronise instead of silently returning
            # nothing.
            self.rebuild_lexical_index()
            if self._lexical.size == 0:
                return outcome

        outcome.dense = self._dense(query)
        outcome.sparse = self._sparse(query)

        if not outcome.dense and not outcome.sparse:
            outcome.reranked = []
            return outcome

        outcome.fused = self._fuse(outcome.dense, outcome.sparse)
        outcome.reranked, outcome.rerank_applied, outcome.rerank_error = self._rerank(
            query, outcome.fused
        )
        outcome.contexts = self._expand(outcome.reranked)
        return outcome

    def rebuild_lexical_index(self) -> int:
        """Recreate the keyword index from active child units in the store."""
        rows = self._store.all_active_children()
        self._lexical.rebuild(rows)
        return self._lexical.size

    def _store_stats_are_empty(self) -> bool:
        """Whether the store genuinely holds no active units.

        Used to distinguish "nothing uploaded yet" from "index out of sync". A
        failure to answer returns ``False`` so the caller resynchronises rather than
        assuming emptiness.
        """
        try:
            stats = self._store.collection_stats()
        except Exception:
            return False
        child_count = stats.get(self._store.child_name, -1)
        return child_count == 0