"""Tests for lexical retrieval and RRF fusion."""

from __future__ import annotations

import pytest

from rag_demo.retrieval.hybrid import rrf_fuse
from rag_demo.retrieval.lexical import LexicalIndex


@pytest.fixture()
def index():
    index = LexicalIndex(user_dictionary=["混合检索", "父子文档"])
    index.rebuild(
        [
            {"id": "a", "text": "混合检索结合了稀疏召回和稠密召回两种方式"},
            {"id": "b", "text": "父子文档让检索用子块而生成用父块"},
            {"id": "c", "text": "Milvus 是一个向量数据库，支持相似度搜索"},
            {"id": "d", "text": "今天天气很好，适合出去散步"},
        ]
    )
    return index


class TestTokenisation:
    def test_chinese_is_segmented_into_words(self, index):
        tokens = index.tokenize("混合检索")

        # cut_for_search emits the whole phrase plus its parts, so a query for a
        # word inside a compound still matches.
        assert "混合" in tokens
        assert "检索" in tokens

    def test_user_dictionary_keeps_compound_together(self, index):
        assert "混合检索" in index.tokenize("混合检索")

    def test_stopwords_are_removed(self, index):
        tokens = index.tokenize("这是一个混合检索的方案")

        assert "的" not in tokens
        assert "是" not in tokens

    def test_latin_words_are_kept_whole(self, index):
        tokens = index.tokenize("Milvus supports BM25 retrieval")

        assert "milvus" in tokens
        assert "bm25" in tokens

    def test_numbers_are_kept(self, index):
        assert "2024" in index.tokenize("published in 2024")

    def test_empty_text_yields_no_tokens(self, index):
        assert index.tokenize("") == []


class TestLexicalSearch:
    def test_chinese_query_matches_shared_terms(self, index):
        hits = index.search("混合检索", limit=3)

        assert hits
        assert hits[0].id == "a"

    def test_unrelated_query_does_not_match(self, index):
        hits = index.search("天气", limit=3)

        assert hits[0].id == "d"

    def test_latin_query_matches(self, index):
        hits = index.search("Milvus", limit=3)

        assert hits[0].id == "c"

    def test_ranking_is_descending_by_score(self, index):
        hits = index.search("检索 文档", limit=4)

        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_limit_is_respected(self, index):
        assert len(index.search("检索 文档 天气 向量", limit=2)) <= 2

    def test_empty_query_returns_nothing(self, index):
        assert index.search("   ", limit=5) == []

    def test_query_on_empty_index_returns_nothing(self):
        empty = LexicalIndex()
        empty.rebuild([])

        assert empty.search("anything", limit=5) == []

    def test_stopword_only_query_returns_nothing(self, index):
        assert index.search("的 了 是", limit=5) == []


class TestRebuild:
    def test_size_reflects_content(self, index):
        assert index.size == 4

    def test_rebuild_replaces_content(self, index):
        index.rebuild([{"id": "z", "text": "only this remains"}])

        assert index.size == 1
        assert index.search("remains", limit=5)[0].id == "z"

    def test_add_appends(self, index):
        index.add([{"id": "e", "text": "新增加的混合检索内容"}])

        assert index.size == 5
        assert any(hit.id == "e" for hit in index.search("新增加", limit=5))

    def test_blank_rows_are_skipped(self, index):
        index.rebuild([{"id": "x", "text": "   "}, {"id": "y", "text": "content"}])

        assert index.size == 1

    def test_text_of_returns_indexed_text(self, index):
        assert "混合检索" in index.text_of("a")

    def test_text_of_unknown_id_is_none(self, index):
        assert index.text_of("missing") is None


class TestRRF:
    def test_found_by_both_channels_outranks_found_by_one(self):
        fused = rrf_fuse([["a", "b"], ["a", "c"]], rrf_k=60)

        assert fused[0][0] == "a"

    def test_scores_are_positive_and_descending(self):
        fused = rrf_fuse([["a", "b", "c"]], rrf_k=60)

        scores = [score for _, score in fused]
        assert all(score > 0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_k_changes_how_much_rank_position_matters(self):
        """A small k exaggerates the advantage of rank 0 over rank 2.

        The ranking is asymmetric on purpose: a symmetric one makes the two ends
        score identically, which hides the effect being tested.
        """
        rankings = [["a", "b", "c", "d"], ["d", "a"]]

        low = dict(rrf_fuse(rankings, rrf_k=1))
        high = dict(rrf_fuse(rankings, rrf_k=100))

        # Ratio of a (rank 0 and rank 1) to d (rank 3 and rank 0).
        low_ratio = low["a"] / low["d"]
        high_ratio = high["a"] / high["d"]
        assert low_ratio > high_ratio

    def test_weights_shift_influence(self):
        fused = rrf_fuse([["a"], ["b"]], rrf_k=60, weights=[10.0, 1.0])

        assert fused[0][0] == "a"

    def test_empty_input_yields_empty_output(self):
        assert rrf_fuse([[], []] , rrf_k=60) == []

    def test_ties_are_broken_deterministically(self):
        first = rrf_fuse([["a"], ["b"]], rrf_k=60)
        second = rrf_fuse([["a"], ["b"]], rrf_k=60)

        # Ties resolve by id, so repeated runs are identical; this matters when
        # comparing retrieval configurations across runs.
        assert first == second

    def test_duplicate_ids_within_one_channel_count_once(self):
        fused = dict(rrf_fuse([["a", "a"]], rrf_k=60))

        # One channel contributes once per document, at its best rank.
        assert fused["a"] == pytest.approx(1 / 61)