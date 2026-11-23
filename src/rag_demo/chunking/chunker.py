"""Two-level chunking: parent context units and child retrieval units.

The shape of the problem:

* Retrieval wants *small* units so a match is specific and a single embedding
  represents one idea.
* Generation wants *large* units so the passage handed to the model still contains
  the reasoning around the sentence that matched.

Splitting into parents and children satisfies both: children are embedded and
searched, parents are what the model reads. Every child knows the parent it came
from, so a hit expands to coherent context.

Splitting strategy is structural first. Markdown headings and PDF paragraph
boundaries already mark topic changes, so they are preferred. Semantic detection —
looking for places where consecutive sentences stop being about the same thing —
is expensive (it embeds every sentence before the chunks are even formed) and is
therefore applied only inside segments that remain oversized after structural
splitting.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from rag_demo.config import ChunkingSettings
from rag_demo.parsers.base import ParsedDocument, SegmentKind, StructuralSegment

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=[.])\s+(?=[A-Z\u4e00-\u9fff])")


@dataclass(slots=True)
class Parent:
    """A unit of context, returned whole to the language model."""

    parent_id: str
    doc_id: str
    text: str
    breadcrumb: str
    page_start: int
    source_name: str
    index: int


@dataclass(slots=True)
class Child:
    """A unit of retrieval, embedded and searched."""

    id: str
    doc_id: str
    parent_id: str
    child_idx: int
    text: str
    breadcrumb: str
    page: int
    source_name: str


@dataclass(slots=True)
class ChunkSet:
    """The result of chunking one document."""

    doc_id: str
    source_name: str
    parents: list[Parent] = field(default_factory=list)
    children: list[Child] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(parent.text) for parent in self.parents)


class Chunker:
    """Split a parsed document into parents, then parents into children."""

    def __init__(self, settings: ChunkingSettings, embedder=None) -> None:
        self._settings = settings
        # Injected only when semantic splitting is needed; keeps the chunker
        # usable (and unit-testable) with no embedding provider at all.
        self._embedder = embedder

    # -------------------------------------------------------------------- public

    def chunk(self, document: ParsedDocument, doc_id: str) -> ChunkSet:
        """Produce the parent/child structure for ``document``."""
        result = ChunkSet(doc_id=doc_id, source_name=document.name)
        groups = self._group_segments(document.segments)

        parent_index = 0
        for group in groups:
            for parent_text, breadcrumb, page in self._split_group_to_parents(group):
                parent_id = f"{doc_id}_p{parent_index}"
                result.parents.append(
                    Parent(
                        parent_id=parent_id,
                        doc_id=doc_id,
                        text=parent_text,
                        breadcrumb=breadcrumb,
                        page_start=page,
                        source_name=document.name,
                        index=parent_index,
                    )
                )
                for child_idx, child_text in enumerate(
                    self._split_parent_to_children(parent_text)
                ):
                    result.children.append(
                        Child(
                            id=f"{parent_id}_c{child_idx}",
                            doc_id=doc_id,
                            parent_id=parent_id,
                            child_idx=child_idx,
                            text=child_text,
                            # The breadcrumb goes with the child because that is
                            # what the reranker sees; without it a fragment like
                            # "as shown in the table below" is unrankable.
                            breadcrumb=breadcrumb,
                            page=page,
                            source_name=document.name,
                        )
                    )
                parent_index += 1

        return result

    # ------------------------------------------------------------------ grouping

    def _group_segments(
        self, segments: list[StructuralSegment]
    ) -> list[list[StructuralSegment]]:
        """Group segments into runs that belong to the same heading.

        Keeping a heading with the text beneath it preserves meaning: a parent that
        starts with its own heading is far more useful than one that starts mid-way
        through a section.
        """
        groups: list[list[StructuralSegment]] = []
        current: list[StructuralSegment] = []

        for segment in segments:
            if segment.kind is SegmentKind.HEADING:
                # A new heading starts a new group, but the heading itself leads it.
                if current:
                    groups.append(current)
                current = [segment]
                continue
            if not current:
                current = [segment]
                continue
            # Start a new group when the heading path changes materially.
            previous_path = current[-1].heading_path
            if segment.heading_path != previous_path and segment.heading_path:
                groups.append(current)
                current = [segment]
            else:
                current.append(segment)

        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------- parents

    def _split_group_to_parents(
        self, group: list[StructuralSegment]
    ) -> list[tuple[str, str, int]]:
        """Turn one group into parent units.

        Returns ``(text, breadcrumb, page)`` triples. Under-sized groups merge so
        that parents are worth retrieving; oversized ones are split, by structure
        where possible and semantically otherwise.
        """
        if not group:
            return []

        max_chars = self._settings.parent_max_chars
        target = self._settings.parent_target_chars
        breadcrumb = " > ".join(
            next(
                (s.heading_path for s in group if s.heading_path),
                (),
            )
        )
        page = next((s.page for s in group if s.page), 0)

        # Accumulate segments until the target size is reached.
        parents: list[tuple[str, str, int]] = []
        buffer: list[str] = []
        buffer_len = 0

        def flush() -> None:
            nonlocal buffer, buffer_len
            text = "\n\n".join(buffer).strip()
            if text:
                parents.append((text, breadcrumb, page))
            buffer = []
            buffer_len = 0

        for segment in group:
            text = segment.text.strip()
            if not text:
                continue

            if len(text) > max_chars:
                # Oversized on its own: flush what is pending, then split it.
                flush()
                for piece in self._split_oversized(text, segment):
                    parents.append((piece, breadcrumb, segment.page or page))
                continue

            if buffer_len + len(text) > max_chars and buffer:
                flush()
            buffer.append(text)
            buffer_len += len(text) + 2  # separator
            if buffer_len >= target:
                flush()

        flush()
        return parents

    def _split_oversized(
        self, text: str, segment: StructuralSegment
    ) -> list[str]:
        """Split text that exceeds the parent maximum, preferring clean boundaries.

        Both structural splitters are tried and scored rather than taking the first
        one that fits. A text with no paragraph breaks still "fits" when split by
        paragraphs, but the cuts land mid-sentence; scoring makes the sentence-aware
        result win.
        """
        max_chars = self._settings.parent_max_chars

        best: list[str] | None = None
        best_score = float("-inf")
        for splitter in (self._split_by_paragraphs, self._split_by_sentences):
            pieces = splitter(text, max_chars)
            if not pieces or any(len(p) > max_chars for p in pieces):
                continue
            score = _boundary_quality(pieces)
            if score > best_score:
                best, best_score = pieces, score

        if best is not None and best_score > 0:
            return best

        if self._settings.semantic_enabled and self._embedder is not None:
            semantic = self._split_semantically(text, max_chars)
            if semantic and all(len(p) <= max_chars for p in semantic):
                return semantic

        if best is not None:
            return best
        return _hard_split(text, max_chars)

    def _split_by_paragraphs(self, text: str, max_chars: int) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return _pack(paragraphs, max_chars)

    def _split_by_sentences(self, text: str, max_chars: int) -> list[str]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]
        return _pack(sentences, max_chars)

    def _split_semantically(self, text: str, max_chars: int) -> list[str]:
        """Split where consecutive sentences stop sharing a topic.

        Consecutive sentences are embedded and the cosine distance between
        neighbours is computed. Boundaries whose distance exceeds a percentile of
        the observed distribution are candidate cut points, strongest first, and
        cuts are taken greedily while respecting ``max_chars``.
        """
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]
        if len(sentences) < 3:
            return []

        try:
            vectors = self._embedder.embed_texts(sentences)
        except Exception:
            # A failing provider must not fail ingestion; fall back to structure.
            return []

        distances: list[float] = []
        for left, right in zip(vectors, vectors[1:], strict=False):
            distances.append(1.0 - _cosine(left, right))

        if not distances:
            return []

        threshold = _percentile(distances, self._settings.semantic_break_percentile)
        candidates = sorted(
            (distance, index)
            for index, distance in enumerate(distances)
            if distance >= threshold
        )
        # Cut at the strongest boundaries first; the set of allowed cuts shrinks
        # as pieces are formed, so ordering matters.
        candidates.reverse()

        boundaries = {index + 1 for _, index in candidates}
        sentences_with_cuts: list[list[str]] = []
        current: list[str] = []
        for index, sentence in enumerate(sentences):
            current.append(sentence)
            if index + 1 in boundaries:
                sentences_with_cuts.append(current)
                current = []
        if current:
            sentences_with_cuts.append(current)

        pieces: list[str] = []
        for run in sentences_with_cuts:
            merged = " ".join(run).strip()
            if len(merged) <= max_chars:
                pieces.append(merged)
            else:
                pieces.extend(_hard_split(merged, max_chars))

        return _merge_short(pieces, max_chars)

    # ------------------------------------------------------------------ children

    def _split_parent_to_children(self, parent_text: str) -> list[str]:
        """Split a parent into overlapping children.

        Overlap exists so a sentence spanning a boundary is fully present in at
        least one child; without it, exact-match retrieval can miss it.
        """
        target = self._settings.child_target_chars
        overlap = self._settings.child_overlap_chars
        if len(parent_text) <= target:
            return [parent_text.strip()]

        children: list[str] = []
        start = 0
        length = len(parent_text)
        while start < length:
            end = min(start + target, length)
            if end < length:
                # Prefer ending at a natural boundary near the target.
                boundary = _find_boundary(parent_text, start, end, target)
                if boundary is not None:
                    end = boundary
            piece = parent_text[start:end].strip()
            if piece:
                children.append(piece)
            if end >= length:
                break
            start = max(end - overlap, start + 1)
        return children or [parent_text.strip()]


def _pack(units: list[str], max_chars: int) -> list[str]:
    """Greedily pack units into strings no longer than ``max_chars``."""
    packed: list[str] = []
    buffer: list[str] = []
    length = 0
    for unit in units:
        if len(unit) > max_chars:
            if buffer:
                packed.append(" ".join(buffer).strip())
                buffer, length = [], 0
            packed.extend(_hard_split(unit, max_chars))
            continue
        if length + len(unit) + 1 > max_chars and buffer:
            packed.append(" ".join(buffer).strip())
            buffer, length = [], 0
        buffer.append(unit)
        length += len(unit) + 1
    if buffer:
        packed.append(" ".join(buffer).strip())
    return [p for p in packed if p]


def _merge_short(pieces: list[str], max_chars: int) -> list[str]:
    """Merge neighbouring pieces that are too short to be worth retrieving."""
    if not pieces:
        return []
    minimum = max(80, max_chars // 6)
    merged: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        if len(merged[-1]) < minimum and len(merged[-1]) + len(piece) <= max_chars:
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)
    return merged


_SENTENCE_ENDINGS = ".!?。！？；;…"

def _boundary_quality(pieces: list[str]) -> float:
    """Score how cleanly a set of pieces ends.

    Each piece that finishes on a sentence terminator scores 1; anything else
    scores 0. A splitter that leaves every piece mid-sentence therefore loses to one
    that cuts between sentences, regardless of which was tried first.
    """
    if not pieces:
        return float("-inf")
    clean = sum(1 for p in pieces[:-1] if p.rstrip() and p.rstrip()[-1] in _SENTENCE_ENDINGS)
    return clean / max(1, len(pieces) - 1)


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split by length as a last resort, preferring whitespace boundaries."""
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = text.rfind(" ", start + max_chars // 2, end)
            if window > start:
                end = window
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def _find_boundary(text: str, start: int, end: int, target: int) -> int | None:
    """Find a sentence or paragraph end near ``end``, searching backwards."""
    lower_bound = max(start + target // 2, start)
    for pattern in ("\n\n", "。", "！", "？", ". ", "! ", "? ", "\n", " "):
        position = text.rfind(pattern, lower_bound, end)
        if position != -1:
            return position + len(pattern)
    return None


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    clamped = min(max(percentile, 0), 100)
    index = int(round((clamped / 100) * (len(ordered) - 1)))
    return ordered[index]


def parent_breadcrumb(parent_text: str) -> str:
    """Derive a short breadcrumb from a parent's first line.

    Used where no heading path exists, such as PDFs without a detected outline.
    """
    first_line = parent_text.strip().split("\n", 1)[0]
    digest = hashlib.sha256(first_line.encode("utf-8")).hexdigest()[:8]
    return f"segment-{digest}"