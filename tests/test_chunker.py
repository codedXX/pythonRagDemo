"""Tests for two-level chunking."""

from __future__ import annotations

import pytest

from rag_demo.chunking.chunker import Chunker, _cosine, _percentile
from rag_demo.config import ChunkingSettings
from rag_demo.parsers.base import ParsedDocument, SegmentKind, StructuralSegment


def _segment(
    text: str,
    kind: SegmentKind = SegmentKind.PARAGRAPH,
    page: int = 1,
    path: tuple[str, ...] = (),
    depth: int = 0,
) -> StructuralSegment:
    return StructuralSegment(
        text=text, kind=kind, page=page, heading_path=path, depth=depth
    )


def _document(*segments: StructuralSegment, name: str = "doc.md") -> ParsedDocument:
    return ParsedDocument(
        name=name,
        segments=list(segments),
        page_count=1,
        char_count=sum(len(s.text) for s in segments),
    )


def _chunker(**overrides) -> Chunker:
    base = dict(
        parent_target_chars=400,
        parent_max_chars=600,
        child_target_chars=150,
        child_overlap_chars=30,
        semantic_enabled=False,
    )
    base.update(overrides)
    return Chunker(ChunkingSettings(**base))


class TestParentChildStructure:
    def test_every_child_has_a_resolvable_parent(self):
        document = _document(_segment("word " * 200))

        result = _chunker().chunk(document, "doc1")

        parent_ids = {p.parent_id for p in result.parents}
        assert result.children
        assert all(child.parent_id in parent_ids for child in result.children)

    def test_every_parent_has_at_least_one_child(self):
        document = _document(_segment("a short paragraph"))

        result = _chunker().chunk(document, "doc1")

        with_children = {child.parent_id for child in result.children}
        assert {p.parent_id for p in result.parents} == with_children

    def test_parents_respect_the_maximum(self):
        document = _document(_segment("sentence. " * 300))

        result = _chunker().chunk(document, "doc1")

        assert result.parents
        assert all(len(p.text) <= 600 for p in result.parents)

    def test_children_respect_the_target_plus_overlap(self):
        document = _document(_segment("word " * 200))

        result = _chunker().chunk(document, "doc1")

        # A child may exceed the target only by the overlap margin.
        assert all(len(c.text) <= 150 + 30 for c in result.children)

    def test_parent_ids_are_unique(self):
        document = _document(*[_segment("paragraph number " * 20) for _ in range(6)])

        result = _chunker().chunk(document, "doc1")

        ids = [p.parent_id for p in result.parents]
        assert len(ids) == len(set(ids))

    def test_child_ids_are_unique(self):
        document = _document(_segment("word " * 400))

        result = _chunker().chunk(document, "doc1")

        ids = [c.id for c in result.children]
        assert len(ids) == len(set(ids))


class TestStructuralPreference:
    def test_headings_become_breadcrumbs(self):
        document = _document(
            _segment("# Chapter One", SegmentKind.HEADING, depth=1, path=("Chapter One",)),
            _segment("Body text under chapter one.", path=("Chapter One",)),
        )

        result = _chunker().chunk(document, "doc1")

        assert result.parents[0].breadcrumb == "Chapter One"

    def test_nested_headings_form_a_path(self):
        document = _document(
            _segment("# Top", SegmentKind.HEADING, depth=1, path=("Top",)),
            _segment("## Sub", SegmentKind.HEADING, depth=2, path=("Top", "Sub")),
            _segment("Detail text.", path=("Top", "Sub")),
        )

        result = _chunker().chunk(document, "doc1")

        breadcrumbs = [p.breadcrumb for p in result.parents]
        assert any(b == "Top > Sub" for b in breadcrumbs)

    def test_children_inherit_the_breadcrumb(self):
        """The breadcrumb must travel with the child, because it is reranker input."""
        document = _document(
            _segment("# Guide", SegmentKind.HEADING, depth=1, path=("Guide",)),
            _segment("word " * 200, path=("Guide",)),
        )

        result = _chunker().chunk(document, "doc1")

        assert result.children
        assert all(c.breadcrumb == "Guide" for c in result.children)

    def test_small_segments_are_merged_into_one_parent(self):
        document = _document(
            _segment("First short paragraph."),
            _segment("Second short paragraph."),
        )

        result = _chunker().chunk(document, "doc1")

        assert len(result.parents) == 1
        assert "First short paragraph" in result.parents[0].text
        assert "Second short paragraph" in result.parents[0].text


class TestOversizedHandling:
    def test_oversized_segment_is_split(self):
        document = _document(_segment("A sentence of reasonable length. " * 100))

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")

        assert len(result.parents) > 1
        assert all(len(p.text) <= 300 for p in result.parents)

    def test_splitting_prefers_sentence_boundaries(self):
        text = ". ".join(f"Sentence number {i}" for i in range(80)) + "."
        document = _document(_segment(text))

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")

        # Pieces should end at a sentence terminator rather than mid-word.
        for parent in result.parents[:-1]:
            assert parent.text.rstrip()[-1] in ".!?。！？"

    def test_no_text_is_lost_when_splitting(self):
        words = [f"w{i}" for i in range(500)]
        document = _document(_segment(" ".join(words)))

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")

        combined = " ".join(p.text for p in result.parents)
        for word in words:
            assert word in combined

    def test_without_screen_text_is_hard_split(self):
        document = _document(_segment("x" * 5000))

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")

        assert all(len(p.text) <= 300 for p in result.parents)


class TestSemanticFallback:
    def test_semantic_detection_not_used_when_structure_suffices(self):
        """The expensive path must stay unused for normal documents."""

        class Forbidden:
            def embed_texts(self, texts):
                raise AssertionError("semantic splitting should not have been used")

        chunker = Chunker(
            ChunkingSettings(
                parent_target_chars=400,
                parent_max_chars=600,
                child_target_chars=150,
                child_overlap_chars=30,
                semantic_enabled=True,
            ),
            embedder=Forbidden(),
        )
        document = _document(_segment("short text"))

        chunker.chunk(document, "doc1")  # must not raise

    def test_semantic_detection_splits_oversized_text(self):
        class FakeEmbedder:
            """Two clearly separated topics, so a boundary must be found."""

            def embed_texts(self, texts):
                vectors = []
                for text in texts:
                    if "cats" in text or "feline" in text:
                        vectors.append([1.0, 0.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0, 0.0])
                return vectors

        chunker = Chunker(
            ChunkingSettings(
                parent_target_chars=200,
                parent_max_chars=200,
                child_target_chars=80,
                child_overlap_chars=10,
                semantic_enabled=True,
                semantic_break_percentile=50,
            ),
            embedder=FakeEmbedder(),
        )
        text = " ".join(["cats are feline animals."] * 6 + ["cars burn petrol."] * 6)
        document = _document(_segment(text))

        result = chunker.chunk(document, "doc1")

        assert len(result.parents) >= 2

    def test_failing_embedder_falls_back_to_structure(self):
        class Broken:
            def embed_texts(self, texts):
                raise RuntimeError("provider down")

        chunker = Chunker(
            ChunkingSettings(
                parent_target_chars=200,
                parent_max_chars=200,
                child_target_chars=80,
                child_overlap_chars=10,
                semantic_enabled=True,
            ),
            embedder=Broken(),
        )
        document = _document(_segment("Sentence about something. " * 40))

        result = chunker.chunk(document, "doc1")

        # Ingestion must still succeed; it just uses structural boundaries.
        assert result.parents


class TestChildSplitting:
    def test_overlap_makes_boundary_sentences_whole(self):
        text = "".join(f"S{i:03d}." for i in range(60))
        document = _document(_segment(text))

        result = _chunker().chunk(document, "doc1")

        joined = [c.text for c in result.children]
        assert len(joined) > 1
        # Consecutive children overlap, so the tail of one appears in the next.
        assert joined[0][-10:] in joined[1]

    def test_short_parent_yields_one_child(self):
        document = _document(_segment("tiny"))

        result = _chunker().chunk(document, "doc1")

        assert len(result.children) == 1
        assert result.children[0].text == "tiny"


class TestHelpers:
    def test_cosine_of_identical_vectors_is_one(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_cosine_of_orthogonal_vectors_is_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_handles_zero_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_percentile_of_empty_is_zero(self):
        assert _percentile([], 90) == 0.0

    def test_percentile_selects_high_value(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert _percentile(values, 100) == 0.5
        assert _percentile(values, 0) == 0.1


def test_empty_document_produces_nothing():
    result = _chunker().chunk(_document(), "doc1")

    assert result.parents == []
    assert result.children == []

def test_paragraph_splitter_does_not_beat_sentence_splitter_when_both_fit():
    """Regression: a text with no blank lines still 'fits' a paragraph split.

    Taking the first splitter that fits cut mid-word ("... Sentence"), which loses
    leading words from the following parent. Pieces must end on sentence ends when
    an equally valid sentence-level split exists.
    """
    text = ". ".join(f"Sentence number {i}" for i in range(80)) + "."
    document = _document(_segment(text))

    result = _chunker(parent_max_chars=300).chunk(document, "doc1")

    for parent in result.parents:
        stripped = parent.text.rstrip()
        assert stripped[-1] in ".!?。！？", f"cut mid-sentence: ...{stripped[-25:]!r}"
