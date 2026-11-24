"""验证父子两级切块的大小、重叠范围和切分边界。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pytest  # 引入测试框架。

from rag_demo.chunking.chunker import Chunker, _cosine, _percentile  # 从 rag_demo.chunking.chunker 引入本模块使用的类型和工具。
from rag_demo.config import ChunkingSettings  # 从 rag_demo.config 导入切块配置。
from rag_demo.parsers.base import ParsedDocument, SegmentKind, StructuralSegment  # 从 rag_demo.parsers.base 引入本模块使用的类型和工具。


def _segment(  # 定义结构片段的处理逻辑。
    text: str,  # 接收文本参数。
    kind: SegmentKind = SegmentKind.PARAGRAPH,  # 接收类别参数。
    page: int = 1,  # 接收页码参数。
    path: tuple[str, ...] = (),  # 接收路径参数。
    depth: int = 0,  # 接收标题层级参数。
) -> StructuralSegment:  # 结束参数声明，进入函数或语句主体。
    return StructuralSegment(  # 返回结构片段的结果。
        text=text, kind=kind, page=page, heading_path=path, depth=depth  # 传入标题层级参数。
    )  # 结束当前多行表达式。


def _document(*segments: StructuralSegment, name: str = "doc.md") -> ParsedDocument:  # 定义文档的处理逻辑。
    return ParsedDocument(  # 返回解析文档的结果。
        name=name,  # 传入名称参数。
        segments=list(segments),  # 传入结构片段列表参数。
        page_count=1,  # 传入总页数参数。
        char_count=sum(len(s.text) for s in segments),  # 传入字符数参数。
    )  # 结束当前多行表达式。


def _chunker(**overrides) -> Chunker:  # 定义测试分块器的处理逻辑。
    base = dict(  # 将基础值设为按字段构造的字典。
        parent_target_chars=400,  # 传入父块目标字符数参数。
        parent_max_chars=600,  # 传入父块字符上限参数。
        child_target_chars=150,  # 传入子块目标字符数参数。
        child_overlap_chars=30,  # 传入相邻子块重叠字符数参数。
        semantic_enabled=False,  # 传入语义切分开关参数。
    )  # 结束当前多行表达式。
    base.update(overrides)  # 用覆盖项更新基础值。
    return Chunker(ChunkingSettings(**base))  # 返回分块器的结果。


class TestParentChildStructure:  # 定义 TestParentChildStructure 类型。
    def test_every_child_has_a_resolvable_parent(self):  # 验证各子块引用的父块均存在。
        document = _document(_segment("word " * 200))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        parent_ids = {p.parent_id for p in result.parents}  # 将parent_ids设为从父块列表中提取的各项。
        assert result.children  # 确认子块列表有效。
        assert all(child.parent_id in parent_ids for child in result.children)  # 确认每一项均满足“parent_ids包含子块的所属父块标识”。

    def test_every_parent_has_at_least_one_child(self):  # 验证每个父块至少有一个子块。
        document = _document(_segment("a short paragraph"))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        with_children = {child.parent_id for child in result.children}  # 将with_children设为从子块列表中提取的各项。
        assert {p.parent_id for p in result.parents} == with_children  # 确认从父块列表中提取的各项等于with_children。

    def test_parents_respect_the_maximum(self):  # 验证父块长度符合配置上限。
        document = _document(_segment("sentence. " * 300))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        assert result.parents  # 确认父块列表有效。
        assert all(len(p.text) <= 600 for p in result.parents)  # 确认每一项均满足“片段的文本的数量不超过600”。

    def test_children_respect_the_target_plus_overlap(self):  # 验证子块长度不超过目标长度与重叠余量之和。
        document = _document(_segment("word " * 200))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        # 子块超出目标长度的部分应落在重叠余量内。
        assert all(len(c.text) <= 150 + 30 for c in result.children)  # 确认每一项均满足“候选项的文本的数量不超过150加上30”。

    def test_parent_ids_are_unique(self):  # 验证各父块标识唯一。
        document = _document(*[_segment("paragraph number " * 20) for _ in range(6)])  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        ids = [p.parent_id for p in result.parents]  # 将标识列表设为从父块列表中提取的各项。
        assert len(ids) == len(set(ids))  # 确认标识列表的数量等于去重集合的数量。

    def test_child_ids_are_unique(self):  # 验证各子块标识唯一。
        document = _document(_segment("word " * 400))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        ids = [c.id for c in result.children]  # 将标识列表设为从子块列表中提取的各项。
        assert len(ids) == len(set(ids))  # 确认标识列表的数量等于去重集合的数量。


class TestStructuralPreference:  # 定义 TestStructuralPreference 类型。
    def test_headings_become_breadcrumbs(self):  # 验证文档标题成为位置路径。
        document = _document(  # 将文档设为文档的结果。
            _segment("# Chapter One", SegmentKind.HEADING, depth=1, path=("Chapter One",)),  # 为结构片段提供输入。
            _segment("Body text under chapter one.", path=("Chapter One",)),  # 为结构片段提供输入。
        )  # 结束当前多行表达式。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        assert result.parents[0].breadcrumb == "Chapter One"  # 确认标题路径等于“Chapter One”。

    def test_nested_headings_form_a_path(self):  # 验证嵌套标题构成层级路径。
        document = _document(  # 将文档设为文档的结果。
            _segment("# Top", SegmentKind.HEADING, depth=1, path=("Top",)),  # 为结构片段提供输入。
            _segment("## Sub", SegmentKind.HEADING, depth=2, path=("Top", "Sub")),  # 为结构片段提供输入。
            _segment("Detail text.", path=("Top", "Sub")),  # 为结构片段提供输入。
        )  # 结束当前多行表达式。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        breadcrumbs = [p.breadcrumb for p in result.parents]  # 将breadcrumbs设为从父块列表中提取的各项。
        assert any(b == "Top > Sub" for b in breadcrumbs)  # 确认至少一项满足“文档长度归一化系数等于“Top > Sub””。

    def test_children_inherit_the_breadcrumb(self):  # 确认子块携带重排所需的标题路径。
        """确认子块携带重排所需的标题路径。"""
        document = _document(  # 将文档设为文档的结果。
            _segment("# Guide", SegmentKind.HEADING, depth=1, path=("Guide",)),  # 为结构片段提供输入。
            _segment("word " * 200, path=("Guide",)),  # 为结构片段提供输入。
        )  # 结束当前多行表达式。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        assert result.children  # 确认子块列表有效。
        assert all(c.breadcrumb == "Guide" for c in result.children)  # 确认每一项均满足“候选项的标题路径等于“Guide””。

    def test_small_segments_are_merged_into_one_parent(self):  # 验证较短的片段可以合并成一个父块。
        document = _document(  # 将文档设为文档的结果。
            _segment("First short paragraph."),  # 为结构片段提供输入。
            _segment("Second short paragraph."),  # 为结构片段提供输入。
        )  # 结束当前多行表达式。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        assert len(result.parents) == 1  # 确认结果的父块列表的数量等于1。
        assert "First short paragraph" in result.parents[0].text  # 确认文本包含“First short paragraph”。
        assert "Second short paragraph" in result.parents[0].text  # 确认文本包含“Second short paragraph”。


class TestOversizedHandling:  # 定义 TestOversizedHandling 类型。
    def test_oversized_segment_is_split(self):  # 验证过长的片段会继续切分。
        document = _document(_segment("A sentence of reasonable length. " * 100))  # 将文档设为文档的结果。

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")  # 将结果设为片段的结果。

        assert len(result.parents) > 1  # 确认结果的父块列表的数量大于1。
        assert all(len(p.text) <= 300 for p in result.parents)  # 确认每一项均满足“片段的文本的数量不超过300”。

    def test_splitting_prefers_sentence_boundaries(self):  # 验证优先在句子边界切分。
        text = ". ".join(f"Sentence number {i}" for i in range(80)) + "."  # 将文本设为拼接后的从指定的下标范围中提取的各项加上“.”。
        document = _document(_segment(text))  # 将文档设为文档的结果。

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")  # 将结果设为片段的结果。

        # 检查片段在句末结束，避免截断单词。
        for parent in result.parents[:-1]:  # 遍历父块列表的指定区间，逐项取得父块。
            assert parent.text.rstrip()[-1] in ".!?。！？"  # 确认“.!?。！？”包含rstrip的结果中对应的项。

    def test_no_text_is_lost_when_splitting(self):  # 验证切分前后的正文完整。
        words = [f"w{i}" for i in range(500)]  # 将words设为从指定的下标范围中提取的各项。
        document = _document(_segment(" ".join(words)))  # 将文档设为文档的结果。

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")  # 将结果设为片段的结果。

        combined = " ".join(p.text for p in result.parents)  # 将combined设为拼接后的从父块列表中提取的各项。
        for word in words:  # 遍历words，逐项取得词条。
            assert word in combined  # 确认词条包含在combined。

    def test_without_screen_text_is_hard_split(self):  # 验证缺少自然分隔符时按长度切分。
        document = _document(_segment("x" * 5000))  # 将文档设为文档的结果。

        result = _chunker(parent_max_chars=300).chunk(document, "doc1")  # 将结果设为片段的结果。

        assert all(len(p.text) <= 300 for p in result.parents)  # 确认每一项均满足“片段的文本的数量不超过300”。


class TestSemanticFallback:  # 定义 TestSemanticFallback 类型。
    def test_semantic_detection_not_used_when_structure_suffices(self):  # 普通文档无需进入成本较高的语义切分流程。
        """普通文档无需进入成本较高的语义切分流程。"""

        class Forbidden:  # 定义 Forbidden 类型。
            def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
                raise AssertionError("semantic splitting should not have been used")  # 向上抛出断言异常。

        chunker = Chunker(  # 将分块器设为分块器的结果。
            ChunkingSettings(  # 为切块配置提供输入。
                parent_target_chars=400,  # 传入父块目标字符数参数。
                parent_max_chars=600,  # 传入父块字符上限参数。
                child_target_chars=150,  # 传入子块目标字符数参数。
                child_overlap_chars=30,  # 传入相邻子块重叠字符数参数。
                semantic_enabled=True,  # 传入语义切分开关参数。
            ),  # 结束当前多行表达式。
            embedder=Forbidden(),  # 传入向量服务参数。
        )  # 结束当前多行表达式。
        document = _document(_segment("short text"))  # 将文档设为文档的结果。

        chunker.chunk(document, "doc1")  # 此调用应正常完成。

    def test_semantic_detection_splits_oversized_text(self):  # 验证过长文本可按语义边界拆分。
        class FakeEmbedder:  # 构造两个差异明显的话题，验证能找出切分边界。
            """构造两个差异明显的话题，验证能找出切分边界。"""

            def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
                vectors = []  # 初始化向量列表为空列表。
                for text in texts:  # 遍历待编码文本，逐项取得文本。
                    if "cats" in text or "feline" in text:  # 当“cats”包含在文本或“feline”包含在文本时进入此分支。
                        vectors.append([1.0, 0.0, 0.0])  # 将列表加入向量列表。
                    else:  # 处理其余情况。
                        vectors.append([0.0, 1.0, 0.0])  # 将列表加入向量列表。
                return vectors  # 返回向量列表。

        chunker = Chunker(  # 将分块器设为分块器的结果。
            ChunkingSettings(  # 为切块配置提供输入。
                parent_target_chars=200,  # 传入父块目标字符数参数。
                parent_max_chars=200,  # 传入父块字符上限参数。
                child_target_chars=80,  # 传入子块目标字符数参数。
                child_overlap_chars=10,  # 传入相邻子块重叠字符数参数。
                semantic_enabled=True,  # 传入语义切分开关参数。
                semantic_break_percentile=50,  # 传入语义距离切分百分位参数。
            ),  # 结束当前多行表达式。
            embedder=FakeEmbedder(),  # 传入向量服务参数。
        )  # 结束当前多行表达式。
        text = " ".join(["cats are feline animals."] * 6 + ["cars burn petrol."] * 6)  # 将文本设为拼接后的列表乘以6加上列表乘以6。
        document = _document(_segment(text))  # 将文档设为文档的结果。

        result = chunker.chunk(document, "doc1")  # 将结果设为片段的结果。

        assert len(result.parents) >= 2  # 确认结果的父块列表的数量不小于2。

    def test_failing_embedder_falls_back_to_structure(self):  # 验证向量服务失败后退回结构切分。
        class Broken:  # 定义 Broken 类型。
            def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
                raise RuntimeError("provider down")  # 向上抛出运行时错误。

        chunker = Chunker(  # 将分块器设为分块器的结果。
            ChunkingSettings(  # 为切块配置提供输入。
                parent_target_chars=200,  # 传入父块目标字符数参数。
                parent_max_chars=200,  # 传入父块字符上限参数。
                child_target_chars=80,  # 传入子块目标字符数参数。
                child_overlap_chars=10,  # 传入相邻子块重叠字符数参数。
                semantic_enabled=True,  # 传入语义切分开关参数。
            ),  # 结束当前多行表达式。
            embedder=Broken(),  # 传入向量服务参数。
        )  # 结束当前多行表达式。
        document = _document(_segment("Sentence about something. " * 40))  # 将文档设为文档的结果。

        result = chunker.chunk(document, "doc1")  # 将结果设为片段的结果。

        # 向量服务发生故障后仍应通过结构切分完成处理。
        assert result.parents  # 确认父块列表有效。


class TestChildSplitting:  # 定义 TestChildSplitting 类型。
    def test_overlap_makes_boundary_sentences_whole(self):  # 验证边界附近的句子可由重叠区域保全。
        text = "".join(f"S{i:03d}." for i in range(60))  # 将文本设为拼接后的从指定的下标范围中提取的各项。
        document = _document(_segment(text))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        joined = [c.text for c in result.children]  # 将拼接结果设为从子块列表中提取的各项。
        assert len(joined) > 1  # 确认拼接结果的数量大于1。
        # 相邻子块重叠，前一块的尾部应出现在下一块内。
        assert joined[0][-10:] in joined[1]  # 确认拼接结果的第二项包含拼接结果中对应的项的指定区间。

    def test_short_parent_yields_one_child(self):  # 验证短父块只生成一个子块。
        document = _document(_segment("tiny"))  # 将文档设为文档的结果。

        result = _chunker().chunk(document, "doc1")  # 将结果设为片段的结果。

        assert len(result.children) == 1  # 确认结果的子块列表的数量等于1。
        assert result.children[0].text == "tiny"  # 确认文本等于“tiny”。


class TestHelpers:  # 定义 TestHelpers 类型。
    def test_cosine_of_identical_vectors_is_one(self):  # 验证相同非零向量的余弦相似度为一。
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)  # 确认计算余弦相似度的结果等于approx的结果。

    def test_cosine_of_orthogonal_vectors_is_zero(self):  # 验证正交向量的余弦相似度为零。
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)  # 确认计算余弦相似度的结果等于approx的结果。

    def test_cosine_handles_zero_vector(self):  # 验证零向量输入得到有效结果。
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # 确认计算余弦相似度的结果等于0.0。

    def test_percentile_of_empty_is_zero(self):  # 验证空序列的百分位值为零。
        assert _percentile([], 90) == 0.0  # 确认计算百分位数的结果等于0.0。

    def test_percentile_selects_high_value(self):  # 验证高百分位得到较大的观测值。
        values = [0.1, 0.2, 0.3, 0.4, 0.5]  # 将值列表设为列表。
        assert _percentile(values, 100) == 0.5  # 确认计算百分位数的结果等于0.5。
        assert _percentile(values, 0) == 0.1  # 确认计算百分位数的结果等于0.1。


def test_empty_document_produces_nothing():  # 验证空文档不会产生分块。
    result = _chunker().chunk(_document(), "doc1")  # 将结果设为片段的结果。

    assert result.parents == []  # 确认父块列表等于空列表。
    assert result.children == []  # 确认子块列表等于空列表。

def test_paragraph_splitter_does_not_beat_sentence_splitter_when_both_fit():  # 回归测试覆盖无空行的连续段落。
    """回归测试覆盖无空行的连续段落。比较分句和分段边界，防止在单词中间截断并丢失后文。"""
    text = ". ".join(f"Sentence number {i}" for i in range(80)) + "."  # 将文本设为拼接后的从指定的下标范围中提取的各项加上“.”。
    document = _document(_segment(text))  # 将文档设为文档的结果。

    result = _chunker(parent_max_chars=300).chunk(document, "doc1")  # 将结果设为片段的结果。

    for parent in result.parents:  # 遍历父块列表，逐项取得父块。
        stripped = parent.text.rstrip()  # 将去除首尾空白的文本设为rstrip的结果。
        assert stripped[-1] in ".!?。！？", f"cut mid-sentence: ...{stripped[-25:]!r}"  # 确认“.!?。！？”包含去除首尾空白的文本中对应的项。
