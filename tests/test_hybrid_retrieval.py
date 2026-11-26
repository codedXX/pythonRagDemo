"""使用真实 Milvus、模拟向量及重排服务验证双路检索，覆盖去重、父块扩展和即时可见性。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import uuid  # 引入唯一标识生成器。

import pytest  # 引入测试框架。

from rag_demo.config import load_settings  # 从 rag_demo.config 导入加载配置。
from rag_demo.embedding.rerank import RerankResult  # 从 rag_demo.embedding.rerank 导入重排结果。
from rag_demo.errors import TransientFailure  # 从 rag_demo.errors 导入暂时性故障。
from rag_demo.retrieval.hybrid import Context, HybridRetriever  # 从 rag_demo.retrieval.hybrid 引入本模块使用的类型和工具。
from rag_demo.retrieval.lexical import LexicalIndex  # 从 rag_demo.retrieval.lexical 导入关键词索引。
from rag_demo.chunking.chunker import Child, Parent
from rag_demo.store.milvus_store import MilvusStore, normalise  # 从 rag_demo.store.milvus_store 引入本模块使用的类型和工具。

pytestmark = pytest.mark.integration  # 将pytestmark设为pytest.mark.integration。

DIM = 6  # 将测试向量维数设为6。


def _vector(seed: int) -> list[float]:  # 生成供测试使用的确定性向量。
    return normalise([((seed + i * 7) % 17) + 1.0 for i in range(DIM)])  # 返回归一化的结果。


class FakeEmbedder:  # 根据内容生成确定性向量，使相关性分数可以复现。
    """根据内容生成确定性向量，使相关性分数可以复现。"""

    def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
        return [_vector(sum(ord(c) for c in text) % 50) for text in texts]  # 返回从待编码文本中提取的各项。


class FakeReranker:  # 按关键词规则模拟重排，使输出顺序可以预期。
    """按关键词规则模拟重排，使输出顺序可以预期。"""

    def __init__(self, keyword: str = "IMPORTANT", failing: bool = False) -> None:  # 保存实例所需的配置和依赖。
        self.keyword = keyword  # 保存传入的keyword。
        self.failing = failing  # 保存传入的failing。
        self.documents: list[str] = []  # 初始化文档为空列表。

    def rerank(self, query, documents, top_n=None):  # 定义重排的处理逻辑。
        self.documents = list(documents)  # 将文档设为文档的列表形式。
        if self.failing:  # 当failing有效时进入此分支。
            raise TransientFailure("rerank", "service unavailable")  # 向上抛出暂时性故障。
        scored = [  # 将已打分的子块标识设为从文档及各项的下标中提取的各项。
            RerankResult(index=i, score=1.0 if self.keyword in document else 0.1)  # 为重排结果提供输入。
            for i, document in enumerate(documents)  # 逐项处理推导式中的i, document。
        ]  # 结束列表。
        scored.sort(key=lambda r: r.score, reverse=True)  # 按指定排序键就地调整已打分的子块标识的顺序。
        return scored  # 返回已打分的子块标识。


def _milvus_available(uri: str) -> bool:  # 探测本机测试数据库是否可连接。
    try:  # 在此捕获后续操作可能产生的异常。
        from pymilvus import MilvusClient  # 从 pymilvus 导入数据库客户端。

        MilvusClient(uri=uri).get_server_version()  # 读取数据库版本。
        return True  # 返回真。
    except Exception:  # 捕获通用异常。
        return False  # 返回假。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def settings(tmp_path_factory):  # 取得当前使用的配置。
    root = tmp_path_factory.mktemp("retrieval")  # 将项目根目录设为mktemp的结果。
    settings = load_settings(root=root, use_env=False)  # 将配置设为加载配置的结果。
    if not _milvus_available(settings.milvus.uri):  # 当不满足“检查测试数据库连接的结果”时进入此分支。
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")  # 跳过缺少外部依赖的测试。
    suffix = uuid.uuid4().hex[:6]  # 将扩展名设为uuid.uuid4().hex的指定区间。
    settings.milvus.child_collection = f"ret_child_{suffix}"  # 将子块集合名设为格式化文本。
    settings.milvus.parent_collection = f"ret_parent_{suffix}"  # 将父块集合名设为格式化文本。
    settings.milvus.meta_collection = f"ret_meta_{suffix}"  # 将元数据集合名设为格式化文本。
    settings.retrieval.dense_top_k = 20  # 将向量召回条数设为20。
    settings.retrieval.sparse_top_k = 20  # 将关键词召回条数设为20。
    settings.retrieval.rerank_candidates = 10  # 将重排候选数量设为10。
    settings.retrieval.context_count = 4  # 将上下文数量设为4。
    settings.retrieval.context_min_count = 2  # 将最少上下文数量设为2。
    settings.retrieval.context_char_budget = 4000  # 将上下文总字符预算设为4000。
    settings.retrieval.parent_excerpt_chars = 500  # 将父块摘录字符数设为500。
    return settings  # 返回配置。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def store(settings):  # 定义向量库的处理逻辑。
    store = MilvusStore(settings)  # 将向量库设为向量库客户端的结果。
    store.drop_all()  # 清理测试集合。
    store.ensure_collections(dimension=DIM)  # 准备集合。
    yield store  # 向调用方逐项提供向量库。
    store.drop_all()  # 清理测试集合。


@pytest.fixture(autouse=True)  # 注册供用例复用的测试夹具。
def clean(store):  # 定义完整边界数的处理逻辑。
    store.client.delete(store.child_name, filter='doc_id != ""')  # 向子块集合名发送删除请求。
    store.client.delete(store.parent_name, filter='doc_id != ""')  # 向父块集合名发送删除请求。
    yield  # 向调用方逐项提供空值。


def _populate(store, doc_id: str, parents: dict[str, tuple[str, str]], children_per_parent: int = 2):  # 写入由给定父块及其子块组成的测试文档。
    """写入由给定父块及其子块组成的测试文档。"""
    child_units = []  # 初始化child_units为空列表。
    parent_units = []  # 初始化parent_units为空列表。
    for index, (parent_id, (parent_text, breadcrumb)) in enumerate(parents.items()):  # 遍历父块列表的键值对及各项的下标，逐项取得组合值。
        parent_units.append(  # 将父块记录的结果加入parent_units。
            Parent(  # 为父块记录提供输入。
                parent_id=parent_id,  # 传入所属父块标识参数。
                doc_id=doc_id,  # 传入文档标识参数。
                text=parent_text,  # 传入文本参数。
                breadcrumb=breadcrumb,  # 传入标题路径参数。
                index=index,
                page_start=index + 1,  # 传入起始页参数。
                source_name=f"{doc_id}.pdf",  # 传入来源文件名参数。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。
        for child_idx in range(children_per_parent):  # 遍历指定的下标范围，逐项取得子块序号。
            text = f"{parent_text} part {child_idx}"  # 将文本设为格式化文本。
            child_units.append(  # 将子块记录的结果加入child_units。
                Child(  # 为子块记录提供输入。
                    id=f"{parent_id}_c{child_idx}",  # 传入标识参数。
                    doc_id=doc_id,  # 传入文档标识参数。
                    parent_id=parent_id,  # 传入所属父块标识参数。
                    child_idx=child_idx,  # 传入子块序号参数。
                    text=text,  # 传入文本参数。
                    breadcrumb=breadcrumb,  # 传入标题路径参数。
                    page=index + 1,  # 传入页码参数。
                    source_name=f"{doc_id}.pdf",  # 传入来源文件名参数。
                )  # 结束当前多行表达式。
            )  # 结束当前多行表达式。
    vectors = [_vector(sum(ord(c) for c in child.text) % 50) for child in child_units]
    store.insert_units(child_units, parent_units, vectors)  # 写入父子块。
    store.activate_document(doc_id)  # 激活文档。


def _retriever(settings, store, reranker=None):  # 定义检索器的处理逻辑。
    lexical = LexicalIndex()  # 将关键词索引设为关键词索引的结果。
    retriever = HybridRetriever(  # 将检索器设为混合检索器的结果。
        settings.retrieval, store, FakeEmbedder(), lexical, reranker  # 为模拟向量编码器提供输入。
    )  # 结束当前多行表达式。
    retriever.rebuild_lexical_index()  # 重建关键词索引。
    return retriever  # 返回检索器。


class TestRecall:  # 定义 TestRecall 类型。
    def test_sparse_channel_finds_keyword_match(self, settings, store):  # 验证关键词通道能召回字面匹配的文档。
        _populate(  # 调用填充。
            store,  # 传入向量库。
            "d1",  # 为本次写入的测试文档指定唯一标识。
            {  # 开始构造字段映射。
                "d1_p0": ("混合检索需要稀疏召回", "ch1"),  # 填入d1_p0字段。
                "d1_p1": ("今天是晴朗的好天气", "ch2"),  # 填入d1_p1字段。
            },  # 结束本组字段。
        )  # 结束当前多行表达式。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("混合检索")  # 将处理结果设为检索的结果。

        assert outcome.sparse  # 确认关键词召回有效。
        assert any("混合检索" in c.text for c in outcome.sparse)  # 确认至少一项满足“候选项的文本包含“混合检索””。

    def test_dense_channel_returns_results(self, settings, store):  # 验证向量通道能返回候选。
        _populate(store, "d1", {"d1_p0": ("some content about retrieval", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("retrieval")  # 将处理结果设为检索的结果。

        assert outcome.dense  # 确认向量召回有效。

    def test_only_active_units_are_recalled(self, settings, store):  # 验证召回结果均已激活。
        _populate(store, "d1", {"d1_p0": ("active content here", "ch1")})  # 调用填充。
        # 写入第二份文档，保留其未激活状态。
        store.insert_units(  # 写入父子块。
            [  # 开始列出各项。
                Child(  # 为子块记录提供输入。
                    id="d2_p0_c0",  # 传入标识参数。
                    doc_id="d2",  # 传入文档标识参数。
                    parent_id="d2_p0",  # 传入所属父块标识参数。
                    child_idx=0,  # 传入子块序号参数。
                    text="inactive content here",  # 传入文本参数。
                    breadcrumb="ch1",  # 传入标题路径参数。
                    page=1,  # 传入页码参数。
                    source_name="d2.pdf",  # 传入来源文件名参数。
                )  # 结束当前多行表达式。
            ],  # 结束列表。
            [  # 开始列出各项。
                Parent(  # 为父块记录提供输入。
                    parent_id="d2_p0",  # 传入所属父块标识参数。
                    doc_id="d2",  # 传入文档标识参数。
                    text="inactive content here",  # 传入文本参数。
                    breadcrumb="ch1",  # 传入标题路径参数。
                    index=0,
                    page_start=1,  # 传入起始页参数。
                    source_name="d2.pdf",  # 传入来源文件名参数。
                )  # 结束当前多行表达式。
            ],  # 结束列表。
            [_vector(9)],
        )  # 结束当前多行表达式。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("inactive content")  # 将处理结果设为检索的结果。

        assert all(c.doc_id != "d2" for c in outcome.fused)  # 确认每一项均满足“候选项的文档标识不等于“d2””。

    def test_one_channel_empty_still_returns_results(self, settings, store):  # 验证其中一路没有命中时仍可检索。
        _populate(store, "d1", {"d1_p0": ("only lexical will match zzz", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        # 只含停用词时，关键词通道没有可用于召回的词项。
        outcome = retriever.retrieve("的 了 是")  # 将处理结果设为检索的结果。

        assert outcome.sparse == []  # 确认关键词召回等于空列表。
        assert outcome.contexts or outcome.fused  # 确认上下文列表或融合结果。

    def test_empty_knowledge_base_returns_nothing(self, settings, store):  # 验证空库查询得到空结果。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("anything")  # 将处理结果设为检索的结果。

        assert outcome.empty  # 确认为空有效。
        assert outcome.contexts == []  # 确认上下文列表等于空列表。


class TestFusion:  # 定义 TestFusion 类型。
    def test_candidate_found_by_both_channels_ranks_first(self, settings, store):  # 验证被两路命中的候选融合后居首。
        _populate(store, "d1", {"d1_p0": ("混合检索 hybrid search explained", "ch1")})  # 调用填充。
        _populate(store, "d2", {"d2_p0": ("unrelated material entirely", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("混合检索")  # 将处理结果设为检索的结果。

        top = outcome.fused[0]  # 将前几项设为融合结果中对应的项。
        assert "dense" in top.channels  # 确认“dense”包含在命中通道。

    def test_fused_positions_are_contiguous(self, settings, store):  # 验证融合排名连续。
        _populate(store, "d1", {"d1_p0": ("alpha beta gamma", "ch1"), "d1_p1": ("delta epsilon", "ch2")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("alpha")  # 将处理结果设为检索的结果。

        assert [c.fused_rank for c in outcome.fused] == list(range(len(outcome.fused)))  # 确认从融合结果中提取的各项等于指定的下标范围的列表形式。


class TestRerank:  # 定义 TestRerank 类型。
    def test_reranker_reorders_candidates(self, settings, store):  # 验证重排模型可以改变候选顺序。
        _populate(  # 调用填充。
            store,  # 传入向量库。
            "d1",  # 为本次写入的测试文档指定唯一标识。
            {  # 开始构造字段映射。
                "d1_p0": ("ordinary content about search", "ch1"),  # 填入d1_p0字段。
                "d1_p1": ("IMPORTANT content about search", "ch2"),  # 填入d1_p1字段。
            },  # 结束本组字段。
        )  # 结束当前多行表达式。
        retriever = _retriever(settings, store, FakeReranker(keyword="IMPORTANT"))  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("search")  # 将处理结果设为检索的结果。

        assert outcome.rerank_applied is True  # 确认是否采用重排为真。
        assert "IMPORTANT" in outcome.contexts[0].text  # 确认文本包含“IMPORTANT”。

    def test_rerank_input_includes_breadcrumb(self, settings, store):  # 重排输入应携带来源上下文，以便公平评价短片段。
        """重排输入应携带来源上下文，以便公平评价短片段。"""
        _populate(store, "d1", {"d1_p0": ("body text", "Chapter Three")})  # 调用填充。
        reranker = FakeReranker()  # 将重排器设为模拟重排器的结果。
        retriever = _retriever(settings, store, reranker)  # 将检索器设为检索器的结果。

        retriever.retrieve("body")  # 检索。

        assert reranker.documents  # 确认文档有效。
        assert all("Chapter Three" in document for document in reranker.documents)  # 确认每一项均满足“文档包含“Chapter Three””。

    def test_more_candidates_are_reranked_than_supplied(self, settings, store):  # 多个子块可能归入一个父块，因此重排候选数需要留出余量。
        """多个子块可能归入一个父块，因此重排候选数需要留出余量。"""
        for index in range(8):  # 遍历指定的下标范围，逐项取得索引。
            _populate(  # 调用填充。
                store,  # 传入向量库。
                f"d{index}",  # 为本次写入的测试文档指定唯一标识。
                {f"d{index}_p0": (f"content number {index} about topic", f"ch{index}")},  # 为填充提供输入。
                children_per_parent=1,  # 传入children_per_parent参数。
            )  # 结束当前多行表达式。
        reranker = FakeReranker()  # 将重排器设为模拟重排器的结果。
        retriever = _retriever(settings, store, reranker)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("content topic")  # 将处理结果设为检索的结果。

        assert len(reranker.documents) > len(outcome.contexts)  # 确认重排器的文档的数量大于处理结果的上下文列表的数量。

    def test_reranker_failure_falls_back_to_fused_order(self, settings, store):  # 验证重排失败后沿用融合顺序。
        _populate(store, "d1", {"d1_p0": ("content that should still be returned", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store, FakeReranker(failing=True))  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("content")  # 将处理结果设为检索的结果。

        assert outcome.rerank_applied is False  # 确认是否采用重排为假。
        assert outcome.rerank_error  # 确认重排失败原因有效。
        # 降级后仍应有足够的上下文回答问题。
        assert outcome.contexts  # 确认上下文列表有效。

    def test_missing_reranker_is_not_an_error(self, settings, store):  # 验证省略重排器时仍可检索。
        _populate(store, "d1", {"d1_p0": ("content here", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store, reranker=None)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("content")  # 将处理结果设为检索的结果。

        assert outcome.rerank_applied is False  # 确认是否采用重排为假。
        assert outcome.contexts  # 确认上下文列表有效。


class TestDeduplication:  # 定义 TestDeduplication 类型。
    def test_siblings_collapse_to_one_context(self, settings, store):  # 验证同一父块的多个子块合并为一条上下文。
        # 构造来自同一父块且都能命中查询的四个子块。
        _populate(store, "d1", {"d1_p0": ("shared topic content", "ch1")}, children_per_parent=4)  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("shared topic content")  # 将处理结果设为检索的结果。

        parent_ids = [c.parent_id for c in outcome.contexts]  # 将parent_ids设为从上下文列表中提取的各项。
        assert len(parent_ids) == len(set(parent_ids)) == 1  # 确认parent_ids的长度等于去重集合的长度等于1。

    def test_supply_is_backfilled_after_collapse(self, settings, store):  # 按父块去重后应从后续候选补足上下文。
        """按父块去重后应从后续候选补足上下文。"""
        # 三个父块各含三个匹配子块，直接截取前三项可能只取到同一父块。
        for index in range(3):  # 遍历指定的下标范围，逐项取得索引。
            _populate(  # 调用填充。
                store,  # 传入向量库。
                f"d{index}",  # 为本次写入的测试文档指定唯一标识。
                {f"d{index}_p0": (f"topic content family {index}", "ch1")},  # 为填充提供输入。
                children_per_parent=3,  # 传入children_per_parent参数。
            )  # 结束当前多行表达式。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("topic content family")  # 将处理结果设为检索的结果。

        assert len(outcome.contexts) >= 2  # 确认处理结果的上下文列表的数量不小于2。
        assert len({c.parent_id for c in outcome.contexts}) == len(outcome.contexts)  # 确认从上下文列表中提取的各项的数量等于处理结果的上下文列表的数量。

    def test_context_count_is_capped(self, settings, store):  # 验证返回的上下文数量符合上限。
        for index in range(10):  # 遍历指定的下标范围，逐项取得索引。
            _populate(store, f"d{index}", {f"d{index}_p0": (f"material {index} shared word", "ch")}, children_per_parent=1)  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("shared word")  # 将处理结果设为检索的结果。

        assert len(outcome.contexts) <= settings.retrieval.context_count  # 确认处理结果的上下文列表的数量不超过上下文数量。


class TestExpansion:  # 定义 TestExpansion 类型。
    def test_context_uses_parent_text_not_child_text(self, settings, store):  # 验证上下文采用父块正文。
        _populate(store, "d1", {"d1_p0": ("the whole parent passage", "ch1")})  # 调用填充。

        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。
        outcome = retriever.retrieve("whole parent")  # 将处理结果设为检索的结果。

        assert "the whole parent passage" in outcome.contexts[0].text  # 确认文本包含“the whole parent passage”。
        assert outcome.contexts[0].matched_text  # 确认命中的子块正文有效。

    def test_matched_child_is_identifiable(self, settings, store):  # 验证可识别实际命中的子块。
        _populate(store, "d1", {"d1_p0": ("parent about检索", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("检索")  # 将处理结果设为检索的结果。

        context = outcome.contexts[0]  # 将上下文设为上下文列表中对应的项。
        assert context.matched_child  # 确认命中的子块标识有效。
        assert context.matched_text  # 确认命中的子块正文有效。

    def test_provenance_is_present(self, settings, store):  # 验证保留文档及位置信息。
        _populate(store, "d1", {"d1_p0": ("source tracked content", "Chapter Two")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("source tracked")  # 将处理结果设为检索的结果。

        context = outcome.contexts[0]  # 将上下文设为上下文列表中对应的项。
        assert context.source_name == "d1.pdf"  # 确认来源文件名等于“d1.pdf”。
        assert context.breadcrumb == "Chapter Two"  # 确认标题路径等于“Chapter Two”。
        assert context.page == 1  # 确认页码等于1。

    def test_oversized_parent_is_excerpted(self, settings, store):  # 验证过长父块只取命中位置附近的摘录。
        long_parent = "x" * 5000  # 将long_parent设为“x”乘以5000。
        _populate(store, "d1", {"d1_p0": (long_parent, "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("x")  # 将处理结果设为检索的结果。

        assert len(outcome.contexts[0].text) <= settings.retrieval.parent_excerpt_chars + 10  # 确认文本的数量不超过父块摘录字符数加上10。

    def test_total_budget_is_respected(self, settings, store):  # 验证各上下文总长度符合预算。
        for index in range(6):  # 遍历指定的下标范围，逐项取得索引。
            _populate(  # 调用填充。
                store,  # 传入向量库。
                f"d{index}",  # 为本次写入的测试文档指定唯一标识。
                {f"d{index}_p0": ("shared budget content " * 60, "ch")},  # 为填充提供输入。
                children_per_parent=1,  # 传入children_per_parent参数。
            )  # 结束当前多行表达式。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("shared budget content")  # 将处理结果设为检索的结果。

        total = sum(len(c.text) for c in outcome.contexts)  # 将总数设为求和的结果。
        assert total <= settings.retrieval.context_char_budget  # 确认总数不超过上下文总字符预算。


class TestDiagnostics:  # 定义 TestDiagnostics 类型。
    def test_all_stages_are_inspectable(self, settings, store):  # 验证每个阶段的结果都可检查。
        _populate(store, "d1", {"d1_p0": ("diagnostic content about search", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store, FakeReranker())  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("diagnostic content")  # 将处理结果设为检索的结果。

        assert outcome.dense is not None  # 确认向量召回不为空值。
        assert outcome.sparse is not None  # 确认关键词召回不为空值。
        assert outcome.fused  # 确认融合结果有效。
        assert outcome.reranked  # 确认重排结果有效。

    def test_channel_membership_is_visible(self, settings, store):  # 验证可查看候选命中的通道。
        _populate(store, "d1", {"d1_p0": ("channel membership visible", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("channel membership")  # 将处理结果设为检索的结果。

        channels = {ch for candidate in outcome.fused for ch in candidate.channels}  # 将命中通道设为从融合结果中提取的各项。
        assert channels <= {"dense", "sparse"}  # 确认命中通道不超过集合。
        assert channels  # 确认命中通道有效。

    def test_rerank_scores_are_recorded(self, settings, store):  # 验证保留重排模型给出的分数。
        _populate(store, "d1", {"d1_p0": ("scored content", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store, FakeReranker())  # 将检索器设为检索器的结果。

        outcome = retriever.retrieve("scored")  # 将处理结果设为检索的结果。

        assert any(c.rerank_score is not None for c in outcome.reranked)  # 确认至少一项满足“候选项的重排分数不为空值”。


class TestLexicalIndexConsistency:  # 定义 TestLexicalIndexConsistency 类型。
    def test_rebuild_picks_up_new_documents(self, settings, store):  # 验证重建后索引包含新增文档。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。
        assert retriever.retrieve("newly added phrase").empty  # 确认为空有效。

        _populate(store, "d1", {"d1_p0": ("newly added phrase here", "ch1")})  # 调用填充。
        assert retriever.rebuild_lexical_index() > 0  # 确认重建关键词索引的结果大于0。

        assert retriever.retrieve("newly added phrase").sparse  # 确认关键词召回有效。

    def test_rebuild_drops_removed_documents(self, settings, store):  # 验证重建后索引移除已删除文档。
        _populate(store, "d1", {"d1_p0": ("temporary content", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。
        assert retriever.retrieve("temporary content").sparse  # 确认关键词召回有效。

        store.delete_document("d1")  # 删除文档。
        retriever.rebuild_lexical_index()  # 重建关键词索引。

        assert retriever.retrieve("temporary content").sparse == []  # 确认关键词召回等于空列表。

    def test_index_reflects_only_active_units(self, settings, store):  # 验证索引只包含已激活的块。
        _populate(store, "d1", {"d1_p0": ("inactive soon content", "ch1")})  # 调用填充。
        retriever = _retriever(settings, store)  # 将检索器设为检索器的结果。

        store.deactivate_document("d1")  # 隐藏文档片段。
        retriever.rebuild_lexical_index()  # 重建关键词索引。

        assert retriever.retrieve("inactive soon").sparse == []  # 确认关键词召回等于空列表。

class TestEmptyKnowledgeBaseBehaviour:  # 定义 TestEmptyKnowledgeBaseBehaviour 类型。
    def test_empty_index_short_circuits_without_embedding(self, settings, store):  # 空库应直接报告缺少内容，并省去不可能命中的向量调用。
        """空库应直接报告缺少内容，并省去不可能命中的向量调用。"""

        class ExplodingEmbedder:  # 定义 ExplodingEmbedder 类型。
            calls = 0  # 将调用记录设为0。

            def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
                ExplodingEmbedder.calls += 1  # 将1累加到调用记录。
                raise AssertionError("no embedding should be attempted when nothing is ingested")  # 向上抛出断言异常。

        lexical = LexicalIndex()  # 将关键词索引设为关键词索引的结果。
        retriever = HybridRetriever(  # 将检索器设为混合检索器的结果。
            settings.retrieval, store, ExplodingEmbedder(), lexical, None  # 为ExplodingEmbedder提供输入。
        )  # 结束当前多行表达式。

        outcome = retriever.retrieve("anything at all")  # 将处理结果设为检索的结果。

        assert outcome.empty  # 确认为空有效。
        assert ExplodingEmbedder.calls == 0  # 确认调用记录等于0。

    def test_stale_index_is_resynchronised_from_the_store(self, settings, store):  # 存储已有文档而关键词索引为空时，应能重建索引并恢复查询。
        """存储已有文档而关键词索引为空时，应能重建索引并恢复查询。"""
        _populate(store, "d1", {"d1_p0": ("recoverable content about search", "ch1")})  # 调用填充。
        lexical = LexicalIndex()  # 刻意保留空索引，以模拟索引过期。
        retriever = HybridRetriever(settings.retrieval, store, FakeEmbedder(), lexical, None)  # 将检索器设为混合检索器的结果。

        outcome = retriever.retrieve("recoverable content")  # 将处理结果设为检索的结果。

        assert outcome.sparse  # 确认关键词召回有效。
        assert not outcome.empty  # 确认为空为空或未启用。
