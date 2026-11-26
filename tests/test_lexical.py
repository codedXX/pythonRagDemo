"""验证关键词检索和倒数排名融合。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pytest  # 引入测试框架。

from rag_demo.retrieval.hybrid import rrf_fuse  # 从 rag_demo.retrieval.hybrid 导入融合各通道排名。
from rag_demo.retrieval.lexical import LexicalIndex  # 从 rag_demo.retrieval.lexical 导入关键词索引。


@pytest.fixture()  # 注册供用例复用的测试夹具。
def index():  # 用已知的中文文本建立可复用的关键词测试索引。
    index = LexicalIndex(user_dictionary=["混合检索", "父子文档"])  # 将索引设为关键词索引的结果。
    index.rebuild(  # 重建关键词索引。
        [  # 开始列出各项。
            {"id": "a", "text": "混合检索结合了稀疏召回和稠密召回两种方式"},  # 为重建关键词索引提供输入。
            {"id": "b", "text": "父子文档让检索用子块而生成用父块"},  # 为重建关键词索引提供输入。
            {"id": "c", "text": "Milvus 是一个向量数据库，支持相似度搜索"},  # 为重建关键词索引提供输入。
            {"id": "d", "text": "今天天气很好，适合出去散步"},  # 为重建关键词索引提供输入。
        ]  # 结束列表。
    )  # 结束当前多行表达式。
    return index  # 返回索引。


class TestTokenisation:  # 定义 TestTokenisation 类型。
    def test_chinese_is_segmented_into_words(self, index):  # 验证中文文本被分成可检索词项。
        tokens = index.tokenize("混合检索")  # 将词元列表设为分词的结果。

        # 搜索模式同时产生复合词与子词，单独查询子词也应命中。
        assert "混合" in tokens  # 确认“混合”包含在词元列表中。
        assert "检索" in tokens  # 确认“检索”包含在词元列表中。

    def test_user_dictionary_keeps_compound_together(self, index):  # 验证用户词典可保留复合词。
        assert "混合检索" in index.tokenize("混合检索")  # 确认“混合检索”包含在分词的结果。

    def test_stopwords_are_removed(self, index):  # 验证分词时过滤停用词。
        tokens = index.tokenize("这是一个混合检索的方案")  # 将词元列表设为分词的结果。

        assert "的" not in tokens  # 确认“的”未包含在词元列表中。
        assert "是" not in tokens  # 确认“是”未包含在词元列表中。

    def test_latin_words_are_kept_whole(self, index):  # 验证拉丁单词保持完整。
        tokens = index.tokenize("Milvus supports BM25 retrieval")  # 将词元列表设为分词的结果。

        assert "milvus" in tokens  # 确认“milvus”包含在词元列表中。
        assert "bm25" in tokens  # 确认“bm25”包含在词元列表中。

    def test_numbers_are_kept(self, index):  # 验证保留数字词项。
        assert "2024" in index.tokenize("published in 2024")  # 确认“2024”包含在分词的结果。

    def test_empty_text_yields_no_tokens(self, index):  # 验证空文本得到空词元序列。
        assert index.tokenize("") == []  # 确认分词的结果等于空列表。


class TestLexicalSearch:  # 定义 TestLexicalSearch 类型。
    def test_chinese_query_matches_shared_terms(self, index):  # 验证中文查询可命中共有词项。
        hits = index.search("混合检索", limit=3)  # 将命中项设为搜索的结果。

        assert hits  # 确认命中项有效。
        assert hits[0].id == "a"  # 确认标识等于“a”。

    def test_unrelated_query_does_not_match(self, index):  # 验证无关查询没有命中。
        hits = index.search("天气", limit=3)  # 将命中项设为搜索的结果。

        assert hits[0].id == "d"  # 确认标识等于“d”。

    def test_latin_query_matches(self, index):  # 验证拉丁文字查询可以命中。
        hits = index.search("Milvus", limit=3)  # 将命中项设为搜索的结果。

        assert hits[0].id == "c"  # 确认标识等于“c”。

    def test_ranking_is_descending_by_score(self, index):  # 验证关键词结果按分数降序排列。
        hits = index.search("检索 文档", limit=4)  # 将命中项设为搜索的结果。

        scores = [hit.score for hit in hits]  # 将分数列表设为从命中项中提取的各项。
        assert scores == sorted(scores, reverse=True)  # 确认分数列表等于按降序排列的分数列表。

    def test_limit_is_respected(self, index):  # 验证结果数受查询上限约束。
        assert len(index.search("检索 文档 天气 向量", limit=2)) <= 2  # 确认搜索的结果的数量不超过2。

    def test_empty_query_returns_nothing(self, index):  # 验证空查询得到空结果。
        assert index.search("   ", limit=5) == []  # 确认搜索的结果等于空列表。

    def test_query_on_empty_index_returns_nothing(self):  # 验证查询空索引得到空结果。
        empty = LexicalIndex()  # 将为空设为关键词索引的结果。
        empty.rebuild([])  # 重建关键词索引。

        assert empty.search("anything", limit=5) == []  # 确认搜索的结果等于空列表。

    def test_stopword_only_query_returns_nothing(self, index):  # 验证只有停用词的查询没有命中。
        assert index.search("的 了 是", limit=5) == []  # 确认搜索的结果等于空列表。


class TestRebuild:  # 定义 TestRebuild 类型。
    def test_size_reflects_content(self, index):  # 验证索引大小反映实际文档数。
        assert index.size == 4  # 确认index.size等于4。

    def test_rebuild_replaces_content(self, index):  # 验证重建索引后替换原有内容。
        index.rebuild([{"id": "z", "text": "only this remains"}])  # 重建关键词索引。

        assert index.size == 1  # 确认index.size等于1。
        assert index.search("remains", limit=5)[0].id == "z"  # 确认标识等于“z”。

    def test_add_appends(self, index):  # 验证追加条目时原索引内容保留。
        index.add([{"id": "e", "text": "新增加的混合检索内容"}])  # 将列表加入索引。

        assert index.size == 5  # 确认index.size等于5。
        assert any(hit.id == "e" for hit in index.search("新增加", limit=5))  # 确认至少一项满足“hit的标识等于“e””。

    def test_blank_rows_are_skipped(self, index):  # 验证建索引时跳过空行。
        index.rebuild([{"id": "x", "text": "   "}, {"id": "y", "text": "content"}])  # 重建关键词索引。

        assert index.size == 1  # 确认index.size等于1。

    def test_text_of_returns_indexed_text(self, index):  # 验证按标识可取回已索引文本。
        assert "混合检索" in index.text_of("a")  # 确认“混合检索”包含在text_of的结果。

    def test_text_of_unknown_id_is_none(self, index):  # 验证未知标识返回空值。
        assert index.text_of("missing") is None  # 确认text_of的结果为空值。


class TestRRF:  # 定义 TestRRF 类型。
    def test_found_by_both_channels_outranks_found_by_one(self):  # 验证双路命中排在单路命中之前。
        fused = rrf_fuse([["a", "b"], ["a", "c"]], rrf_k=60)  # 将融合结果设为融合各通道排名的结果。

        assert fused[0][0] == "a"  # 确认融合结果的首项的首项等于“a”。

    def test_scores_are_positive_and_descending(self):  # 验证融合分数为正且降序排列。
        fused = rrf_fuse([["a", "b", "c"]], rrf_k=60)  # 将融合结果设为融合各通道排名的结果。

        scores = [score for _, score in fused]  # 将分数列表设为从融合结果中提取的各项。
        assert all(score > 0 for score in scores)  # 确认每一项均满足“分数大于0”。
        assert scores == sorted(scores, reverse=True)  # 确认分数列表等于按降序排列的分数列表。

    def test_k_changes_how_much_rank_position_matters(self):  # 使用较小的融合常数放大名次差距。
        """使用较小的融合常数放大名次差距；故意使用不对称排名，让两端的分数有别。"""
        rankings = [["a", "b", "c", "d"], ["d", "a"]]  # 将rankings设为列表。

        low = dict(rrf_fuse(rankings, rrf_k=1))  # 将low设为融合各通道排名的结果的字典形式。
        high = dict(rrf_fuse(rankings, rrf_k=100))  # 将high设为融合各通道排名的结果的字典形式。

        # 比较两个候选在两路排名不同时的得分比。
        low_ratio = low["a"] / low["d"]  # 将low_ratio设为low中的a除以low中的d。
        high_ratio = high["a"] / high["d"]  # 将high_ratio设为high中的a除以high中的d。
        assert low_ratio > high_ratio  # 确认low_ratio大于high_ratio。

    def test_weights_shift_influence(self):  # 验证调整通道权重会改变融合结果。
        fused = rrf_fuse([["a"], ["b"]], rrf_k=60, weights=[10.0, 1.0])  # 将融合结果设为融合各通道排名的结果。

        assert fused[0][0] == "a"  # 确认融合结果的首项的首项等于“a”。

    def test_empty_input_yields_empty_output(self):  # 验证没有输入时输出也为空。
        assert rrf_fuse([[], []] , rrf_k=60) == []  # 确认融合各通道排名的结果等于空列表。

    def test_ties_are_broken_deterministically(self):  # 验证同分候选的顺序稳定。
        first = rrf_fuse([["a"], ["b"]], rrf_k=60)  # 将首项设为融合各通道排名的结果。
        second = rrf_fuse([["a"], ["b"]], rrf_k=60)  # 将第二项设为融合各通道排名的结果。

        # 同分时以标识决定顺序，以便多次运行和对比配置。
        assert first == second  # 确认首项等于第二项。

    def test_duplicate_ids_within_one_channel_count_once(self):  # 验证单个通道内的重复标识只计分一次。
        fused = dict(rrf_fuse([["a", "a"]], rrf_k=60))  # 将融合结果设为融合各通道排名的结果的字典形式。

        # 单个通道对同一候选只按最高名次计分一次。
        assert fused["a"] == pytest.approx(1 / 61)  # 确认融合结果中的a等于approx的结果。

def test_fusion_ranks_and_partial_rerank_preserve_unscored_candidates():  # 验证融合名次和重排仅返回部分分数时的候选保留顺序。
    from unittest.mock import Mock  # 用替身指定重排服务的响应。
    from rag_demo.config import RetrievalSettings  # 使用检索配置的默认融合参数。
    from rag_demo.embedding.rerank import RerankResult  # 构造带输入位置和分数的重排结果。
    from rag_demo.retrieval.hybrid import Candidate, HybridRetriever  # 引入候选类型及融合、重排逻辑。

    def candidate(identifier, **ranks):  # 为不同召回路径构造最小候选样本。
        return Candidate(identifier, "parent", "text", "", 1, "source", **ranks)  # 固定正文与来源，仅按用例指定标识和名次。

    settings = RetrievalSettings()  # 创建本用例的检索配置。
    reranker = Mock()  # 以替身控制哪些候选获得重排分数。
    retriever = HybridRetriever(settings, None, None, None, reranker)  # 构造仅用于测试融合与重排的检索器。
    fused = retriever._fuse(  # 将两路召回的排名送入融合。
        [candidate("shared", dense_rank=0), candidate("a", dense_rank=1)],  # 稠密通道包含共同命中项和独有项 a。
        [candidate("shared", sparse_rank=0), candidate("b", sparse_rank=1)],  # 稀疏通道包含共同命中项和独有项 b。
    )  # 取得融合候选列表。
    assert [(c.id, c.dense_rank, c.sparse_rank, c.fused_rank) for c in fused] == [  # 同时核对两路原始名次和融合后的名次。
        ("shared", 0, 0, 0), ("a", 1, None, 1), ("b", None, 1, 2)  # 共同命中项排首位，两路独有项保留各自名次。
    ]  # 完成融合位置的核对。
    assert [c.fused_score for c in fused] == pytest.approx([2/61, 1/62, 1/62])  # 按倒数排名融合公式检查每个候选的得分。
    reranker.rerank.return_value = [RerankResult(index=2, score=0.9)]  # 让重排服务只给第三个候选打分。
    result, applied, error = retriever._rerank("query", fused)  # 执行部分重排并取出结果及状态。
    assert applied and error is None  # 确认部分结果仍视为重排成功。
    assert [(c.id, c.rerank_rank) for c in result] == [("b", 0), ("shared", None), ("a", None)]  # 确认有分数的 b 前移，未评分项保留原有相对顺序。
