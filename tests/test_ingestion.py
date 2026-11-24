"""借助真实 Milvus 和模拟向量服务检验入库的原子性、即时可见性及同名文件替换。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import uuid  # 引入唯一标识生成器。

import pymupdf  # 引入版面解析库。
import pytest  # 引入测试框架。

from rag_demo.chunking.chunker import Chunker  # 从 rag_demo.chunking.chunker 导入分块器。
from rag_demo.config import load_settings  # 从 rag_demo.config 导入加载配置。
from rag_demo.ingestion import IngestionError, IngestionPipeline, Stage, remove_document  # 从 rag_demo.ingestion 引入本模块使用的类型和工具。
from rag_demo.store.milvus_store import MilvusStore  # 从 rag_demo.store.milvus_store 导入向量库客户端。
from rag_demo.store.registry import DocumentRegistry, DocumentState  # 从 rag_demo.store.registry 引入本模块使用的类型和工具。

pytestmark = pytest.mark.integration  # 将pytestmark设为pytest.mark.integration。

DIM = 8  # 将测试向量维数设为8。


class FakeEmbedder:  # 生成确定性向量，使测试无需密钥且分数可重复。
    """生成确定性向量，使测试无需密钥且分数可重复。"""

    def __init__(self, fail_on: str | None = None) -> None:  # 保存实例所需的配置和依赖。
        self.calls = 0  # 将调用记录设为0。
        self.fail_on = fail_on  # 保存传入的fail_on。

    def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
        self.calls += 1  # 将1累加到调用记录。
        if self.fail_on and self.fail_on in " ".join(texts):  # 当fail_on且fail_on包含在拼接后的待编码文本时进入此分支。
            raise RuntimeError("provider exploded")  # 向上抛出运行时错误。
        return [self._vector(text) for text in texts]  # 返回从待编码文本中提取的各项。

    @staticmethod  # 声明无需实例状态的静态方法。
    def _vector(text: str) -> list[float]:  # 生成供测试使用的确定性向量。
        # 按内容生成向量，相同文本始终得到相同的坐标。
        seed = sum(ord(c) for c in text) % 97  # 将种子设为求和的结果对其取模97。
        return [((seed + i) % 13) / 13.0 for i in range(DIM)]  # 返回从指定的下标范围中提取的各项。

    def probe_dimension(self, text: str = "ping") -> int:  # 定义探测向量维度的处理逻辑。
        return DIM  # 返回测试向量维数。


def _milvus_available(uri: str) -> bool:  # 探测本机测试数据库是否可连接。
    try:  # 在此捕获后续操作可能产生的异常。
        from pymilvus import MilvusClient  # 从 pymilvus 导入数据库客户端。

        MilvusClient(uri=uri).get_server_version()  # 读取数据库版本。
        return True  # 返回真。
    except Exception:  # 捕获通用异常。
        return False  # 返回假。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def settings(tmp_path_factory):  # 取得当前使用的配置。
    root = tmp_path_factory.mktemp("ingest")  # 将项目根目录设为mktemp的结果。
    settings = load_settings(root=root, use_env=False)  # 将配置设为加载配置的结果。
    if not _milvus_available(settings.milvus.uri):  # 当不满足“检查测试数据库连接的结果”时进入此分支。
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")  # 跳过缺少外部依赖的测试。
    suffix = uuid.uuid4().hex[:6]  # 将扩展名设为uuid.uuid4().hex的指定区间。
    settings.milvus.child_collection = f"ing_child_{suffix}"  # 将子块集合名设为格式化文本。
    settings.milvus.parent_collection = f"ing_parent_{suffix}"  # 将父块集合名设为格式化文本。
    settings.milvus.meta_collection = f"ing_meta_{suffix}"  # 将元数据集合名设为格式化文本。
    settings.chunking.parent_target_chars = 300  # 将父块目标字符数设为300。
    settings.chunking.parent_max_chars = 400  # 将父块字符上限设为400。
    settings.chunking.child_target_chars = 120  # 将子块目标字符数设为120。
    settings.chunking.child_overlap_chars = 20  # 将相邻子块重叠字符数设为20。
    settings.chunking.semantic_enabled = False  # 将语义切分开关设为假。
    return settings  # 返回配置。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def store(settings):  # 定义向量库的处理逻辑。
    store = MilvusStore(settings)  # 将向量库设为向量库客户端的结果。
    store.drop_all()  # 清理测试集合。
    store.ensure_collections(dimension=DIM)  # 准备集合。
    yield store  # 向调用方逐项提供向量库。
    store.drop_all()  # 清理测试集合。


@pytest.fixture()  # 注册供用例复用的测试夹具。
def registry(tmp_path):  # 定义登记表的处理逻辑。
    return DocumentRegistry(tmp_path / "rag.db")  # 返回文档登记表的结果。


@pytest.fixture()  # 注册供用例复用的测试夹具。
def pipeline(settings, store, registry):  # 定义入库流程的处理逻辑。
    embedder = FakeEmbedder()  # 将向量服务设为模拟向量编码器的结果。
    return IngestionPipeline(  # 返回入库流程的结果。
        settings,  # 传入配置。
        store,  # 传入向量库。
        registry,  # 传入登记表。
        embedder,  # 传入向量服务。
        Chunker(settings.chunking, embedder=embedder),  # 为分块器提供输入。
    )  # 结束当前多行表达式。


def _markdown(body: str) -> bytes:  # 定义标记文档的处理逻辑。
    return body.encode("utf-8")  # 返回文本编码的结果。


def _pdf(pages: list[str]) -> bytes:  # 定义测试文档的处理逻辑。
    document = pymupdf.open()  # 将文档设为打开文档的结果。
    for body in pages:  # 遍历页面列表，逐项取得正文。
        page = document.new_page()  # 将页码设为创建页面的结果。
        page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), body, fontsize=11)  # 写入文本框。
    data = document.tobytes()  # 将数据设为导出文档字节串的结果。
    document.close()  # 关闭文档并释放资源。
    return data  # 返回数据。


def _clean(store, registry):  # 定义完整边界数的处理逻辑。
    store.client.delete(store.child_name, filter='doc_id != ""')  # 向子块集合名发送删除请求。
    store.client.delete(store.parent_name, filter='doc_id != ""')  # 向父块集合名发送删除请求。
    for record in registry.list_documents():  # 遍历列出登记文档的结果，逐项取得登记记录。
        registry.remove(record.doc_id)  # 移除。


class TestHappyPath:  # 定义 TestHappyPath 类型。
    def test_markdown_ingestion_produces_retrievable_units(self, pipeline, store, registry):  # 验证标记文档入库后产生可检索的块。
        body = "# Section\n\n" + "\n\n".join(f"Paragraph {i} with meaningful content." for i in range(6))  # 将正文设为文本内容加上拼接后的从指定的下标范围中提取的各项。

        result = pipeline.ingest("notes.md", _markdown(body))  # 将结果设为入库的结果。

        assert result.parent_count >= 1  # 确认父块数至少为1。
        assert result.child_count >= result.parent_count  # 确认子块数至少为父块数。
        record = registry.get(result.doc_id)  # 将登记记录设为读取的结果。
        assert record is not None  # 确认登记记录不为空值。
        assert record.state is DocumentState.ACTIVE  # 确认状态为DocumentState.ACTIVE。
        assert record.child_count == result.child_count  # 确认子块数等于子块数。
        _clean(store, registry)  # 调用完整边界数。

    def test_pdf_ingestion_works(self, pipeline, store, registry):  # 验证便携文档可以成功入库。
        body = "A substantial paragraph of text for ingestion. " * 12  # 将正文设为文本内容乘以12。

        result = pipeline.ingest("paper.pdf", _pdf([body, body]))  # 将结果设为入库的结果。

        assert result.child_count > 0  # 确认子块数大于0。
        _clean(store, registry)  # 调用完整边界数。

    def test_units_are_retrievable_immediately(self, pipeline, store, registry):  # 在整条入库流程上验证强一致性。
        """在整条入库流程上验证强一致性。"""
        vector = FakeEmbedder._vector("hello world")  # 将向量设为测试向量的结果。
        body = "hello world " * 40  # 将正文设为“hello world ”乘以40。

        pipeline.ingest("immediate.md", _markdown(body))  # 入库。

        hits = store.search_children(vector, limit=5)  # 将命中项设为检索子块的结果。
        assert hits  # 确认命中项有效。
        _clean(store, registry)  # 调用完整边界数。

    def test_progress_is_reported_in_order(self, pipeline, store, registry):  # 验证按入库顺序报告进度。
        seen: list[str] = []  # 初始化已处理项为空列表。

        pipeline.ingest(  # 入库。
            "progress.md",  # 给用于观察进度的上传文档命名。
            _markdown("content " * 100),  # 为标记文档提供输入。
            on_progress=lambda p: seen.append(p.stage.value),  # 传入进度回调参数。
        )  # 结束当前多行表达式。

        assert seen[0] == Stage.PARSING.value  # 确认已见标识的首项等于值。
        assert Stage.CHUNKING.value in seen  # 确认值包含在已处理项。
        assert Stage.EMBEDDING.value in seen  # 确认值包含在已处理项。
        assert Stage.PERSISTING.value in seen  # 确认值包含在已处理项。
        assert seen[-1] == Stage.DONE.value  # 确认已见标识中对应的项等于值。
        _clean(store, registry)  # 调用完整边界数。

    def test_char_count_is_recorded(self, pipeline, store, registry):  # 验证登记表记录文档字符数。
        result = pipeline.ingest("count.md", _markdown("word " * 200))  # 将结果设为入库的结果。

        assert result.char_count > 0  # 确认字符数大于0。
        assert registry.get(result.doc_id).char_count == result.char_count  # 确认字符数等于字符数。
        _clean(store, registry)  # 调用完整边界数。


class TestAtomicity:  # 定义 TestAtomicity 类型。
    def test_failed_ingestion_leaves_nothing_retrievable(self, settings, store, registry):  # 向量计算失败后，不应留下可检索的半成品。
        """向量计算失败后，不应留下可检索的半成品。"""
        embedder = FakeEmbedder(fail_on="EXPLODE")  # 将向量服务设为模拟向量编码器的结果。
        pipeline = IngestionPipeline(  # 将入库流程设为入库流程的结果。
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)  # 为分块器提供输入。
        )  # 结束当前多行表达式。

        with pytest.raises(IngestionError):  # 验证下方操作会抛出入库异常。
            pipeline.ingest("broken.md", _markdown("EXPLODE " * 100))  # 入库。

        assert store.search_children(FakeEmbedder._vector("EXPLODE"), limit=10) == []  # 确认检索子块的结果等于空列表。
        assert store.all_active_children() == []  # 确认读取有效子块的结果等于空列表。

    def test_failure_is_recorded_with_its_stage(self, settings, store, registry):  # 验证失败记录带有所处阶段。
        embedder = FakeEmbedder(fail_on="EXPLODE")  # 将向量服务设为模拟向量编码器的结果。
        pipeline = IngestionPipeline(  # 将入库流程设为入库流程的结果。
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)  # 为分块器提供输入。
        )  # 结束当前多行表达式。

        with pytest.raises(IngestionError) as excinfo:  # 验证下方操作会抛出入库异常。
            pipeline.ingest("broken.md", _markdown("EXPLODE " * 100))  # 入库。

        assert excinfo.value.stage is Stage.EMBEDDING  # 确认阶段为Stage.EMBEDDING。
        records = registry.list_documents()  # 将登记记录列表设为列出登记文档的结果。
        assert records[0].state is DocumentState.FAILED  # 确认状态为DocumentState.FAILED。
        assert Stage.EMBEDDING.value in (records[0].error or "")  # 确认错误或空字符串包含值。

    def test_scanned_pdf_fails_before_any_embedding_call(  # 解析失败时应省去后续的向量调用。
        self, settings, store, registry  # 组合当前结果所需的各项数据。
    ):  # 结束参数声明，进入函数或语句主体。
        """解析失败时应省去后续的向量调用。"""
        embedder = FakeEmbedder()  # 将向量服务设为模拟向量编码器的结果。
        pipeline = IngestionPipeline(  # 将入库流程设为入库流程的结果。
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)  # 为分块器提供输入。
        )  # 结束当前多行表达式。
        blank = pymupdf.open()  # 将blank设为打开文档的结果。
        blank.new_page()  # 创建页面。
        content = blank.tobytes()  # 将内容设为导出文档字节串的结果。
        blank.close()  # 关闭blank并释放资源。

        with pytest.raises(IngestionError) as excinfo:  # 验证下方操作会抛出入库异常。
            pipeline.ingest("scan.pdf", content)  # 入库。

        assert excinfo.value.stage is Stage.PARSING  # 确认阶段为Stage.PARSING。
        assert embedder.calls == 0  # 确认调用记录等于0。

    def test_unsupported_type_fails_without_embedding(self, settings, store, registry):  # 验证不支持的格式不会触发向量调用。
        embedder = FakeEmbedder()  # 将向量服务设为模拟向量编码器的结果。
        pipeline = IngestionPipeline(  # 将入库流程设为入库流程的结果。
            settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)  # 为分块器提供输入。
        )  # 结束当前多行表达式。

        with pytest.raises(IngestionError) as excinfo:  # 验证下方操作会抛出入库异常。
            pipeline.ingest("notes.txt", b"plain text")  # 入库。

        assert excinfo.value.stage is Stage.PARSING  # 确认阶段为Stage.PARSING。
        assert embedder.calls == 0  # 确认调用记录等于0。


class TestReplacement:  # 定义 TestReplacement 类型。
    def test_reuploading_same_name_replaces_rather_than_duplicates(  # 验证上传同名文件会替换旧版本。
        self, pipeline, store, registry  # 组合当前结果所需的各项数据。
    ):  # 结束参数声明，进入函数或语句主体。
        pipeline.ingest("same.md", _markdown("original content " * 40))  # 入库。
        pipeline.ingest("same.md", _markdown("replacement content " * 40))  # 入库。

        active = store.all_active_children()  # 将激活状态设为读取有效子块的结果。

        assert active  # 确认激活状态有效。
        assert all("replacement" in row["text"] for row in active)  # 确认每一项均满足“记录中的文本包含“replacement””。
        assert len([r for r in registry.list_usable() if r.name == "same.md"]) == 1  # 确认从列出有效文档的结果中提取的各项的数量等于1。
        _clean(store, registry)  # 调用完整边界数。

    def test_changed_content_gets_a_new_document_id(self, pipeline, store, registry):  # 验证内容改变后文档使用新标识。
        first = pipeline.ingest("doc.md", _markdown("version one " * 40))  # 将首项设为入库的结果。
        second = pipeline.ingest("doc.md", _markdown("version two " * 40))  # 将第二项设为入库的结果。

        # 文件内容改变会产生新标识，旧版本则按文件名清除。
        assert first.doc_id != second.doc_id  # 确认文档标识不等于文档标识。
        assert len(registry.list_usable()) == 1  # 确认列出有效文档的结果的数量等于1。
        _clean(store, registry)  # 调用完整边界数。


class TestRemoval:  # 定义 TestRemoval 类型。
    def test_remove_document_deletes_content_and_record(self, pipeline, store, registry):  # 验证删除同时清理登记行和片段。
        result = pipeline.ingest("gone.md", _markdown("content to remove " * 40))  # 将结果设为入库的结果。

        assert remove_document(result.doc_id, store, registry) is True  # 确认清除文档及片段的结果为真。

        assert store.search_children(FakeEmbedder._vector("content"), limit=10) == []  # 确认检索子块的结果等于空列表。
        assert registry.get(result.doc_id) is None  # 确认读取的结果为空值。

    def test_remove_unknown_document_reports_false(self, store, registry):  # 验证删除不存在的文档返回假值。
        assert remove_document("does-not-exist", store, registry) is False  # 确认清除文档及片段的结果为假。