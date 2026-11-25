"""通过真实 Milvus 和模拟模型验证从上传到回答的完整流程，覆盖切块、写入、检索以及来源传递。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。
import uuid  # 引入唯一标识生成器。

import pytest  # 引入测试框架。
from fastapi.testclient import TestClient  # 从 fastapi.testclient 导入测试客户端。

from rag_demo.api.app import create_app  # 从 rag_demo.api.app 导入创建应用。
from rag_demo.api.services import Services  # 从 rag_demo.api.services 导入应用服务集合。
from rag_demo.chunking.chunker import Chunker  # 从 rag_demo.chunking.chunker 导入分块器。
from rag_demo.config import load_settings  # 从 rag_demo.config 导入加载配置。
from rag_demo.embedding.rerank import RerankResult  # 从 rag_demo.embedding.rerank 导入重排结果。
from rag_demo.generation.answer import Answerer  # 从 rag_demo.generation.answer 导入问答服务。
from rag_demo.ingestion import IngestionPipeline  # 从 rag_demo.ingestion 导入入库流程。
from rag_demo.retrieval.hybrid import HybridRetriever  # 从 rag_demo.retrieval.hybrid 导入混合检索器。
from rag_demo.retrieval.lexical import LexicalIndex  # 从 rag_demo.retrieval.lexical 导入关键词索引。
from rag_demo.store.milvus_store import MilvusStore, normalise  # 从 rag_demo.store.milvus_store 引入本模块使用的类型和工具。
from rag_demo.store.registry import DocumentRegistry  # 从 rag_demo.store.registry 导入文档登记表。

pytestmark = pytest.mark.integration  # 将pytestmark设为pytest.mark.integration。

DIM = 10  # 将测试向量维数设为10。


class HashEmbedder:  # 定义 HashEmbedder 类型。
    def embed_texts(self, texts):  # 定义批量生成文本向量的处理逻辑。
        return [  # 返回从待编码文本中提取的各项。
            normalise(  # 为归一化提供输入。
                [((sum(ord(c) for c in text) + i * 7) % 31) + 1.0 for i in range(DIM)]  # 为ord提供输入。
            )  # 结束当前多行表达式。
            for text in texts  # 逐项处理推导式中的文本。
        ]  # 结束列表。

    def probe_dimension(self, text: str = "ping") -> int:  # 定义探测向量维度的处理逻辑。
        return DIM  # 返回测试向量维数。


class KeywordReranker:  # 定义 KeywordReranker 类型。
    def rerank(self, query, documents, top_n=None):  # 定义重排的处理逻辑。
        scored = [  # 将已打分的子块标识设为从文档及各项的下标中提取的各项。
            RerankResult(index=i, score=1.0 if query[:2] in document else 0.3)  # 为重排结果提供输入。
            for i, document in enumerate(documents)  # 逐项处理推导式中的i, document。
        ]  # 结束列表。
        scored.sort(key=lambda r: r.score, reverse=True)  # 按指定排序键就地调整已打分的子块标识的顺序。
        return scored  # 返回已打分的子块标识。


class EchoGenerator:  # 生成带有传入来源名称的模拟回答。
    """生成带有传入来源名称的模拟回答。"""

    def __init__(self) -> None:  # 保存实例所需的配置和依赖。
        self.prompts: list[str] = []  # 初始化提示消息记录为空列表。

    def stream_chat(self, messages):  # 定义流式生成回复的处理逻辑。
        self.prompts.append(messages[-1].content)  # 将内容加入提示消息记录。
        for piece in ("根据", "提供的", "资料，", "答案是", "这样。"):  # 遍历组合值，逐项取得片段。
            yield piece  # 向调用方逐项提供片段。

    def complete(self, messages):  # 定义获取完整回答的处理逻辑。
        return "改写后的问题"  # 返回“改写后的问题”。


def _milvus_available(uri: str) -> bool:  # 探测本机测试数据库是否可连接。
    try:  # 在此捕获后续操作可能产生的异常。
        from pymilvus import MilvusClient  # 从 pymilvus 导入数据库客户端。

        MilvusClient(uri=uri).get_server_version()  # 读取数据库版本。
        return True  # 返回真。
    except Exception:  # 捕获通用异常。
        return False  # 返回假。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def settings(tmp_path_factory):  # 取得当前使用的配置。
    root = tmp_path_factory.mktemp("e2e")  # 将项目根目录设为mktemp的结果。
    settings = load_settings(root=root, use_env=False)  # 将配置设为加载配置的结果。
    if not _milvus_available(settings.milvus.uri):  # 当不满足“检查测试数据库连接的结果”时进入此分支。
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")  # 跳过缺少外部依赖的测试。
    suffix = uuid.uuid4().hex[:6]  # 将扩展名设为uuid.uuid4().hex的指定区间。
    settings.milvus.child_collection = f"e2e_child_{suffix}"  # 将子块集合名设为格式化文本。
    settings.milvus.parent_collection = f"e2e_parent_{suffix}"  # 将父块集合名设为格式化文本。
    settings.milvus.meta_collection = f"e2e_meta_{suffix}"  # 将元数据集合名设为格式化文本。
    settings.chunking.parent_target_chars = 400  # 将父块目标字符数设为400。
    settings.chunking.parent_max_chars = 600  # 将父块字符上限设为600。
    settings.chunking.child_target_chars = 180  # 将子块目标字符数设为180。
    settings.chunking.child_overlap_chars = 30  # 将相邻子块重叠字符数设为30。
    settings.chunking.semantic_enabled = False  # 将语义切分开关设为假。
    settings.retrieval.context_count = 4  # 将上下文数量设为4。
    settings.retrieval.rerank_candidates = 10  # 将重排候选数量设为10。
    return settings  # 返回配置。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def store(settings):  # 定义向量库的处理逻辑。
    store = MilvusStore(settings)  # 将向量库设为向量库客户端的结果。
    store.drop_all()  # 清理测试集合。
    store.ensure_collections(dimension=DIM)  # 准备集合。
    yield store  # 向调用方逐项提供向量库。
    store.drop_all()  # 清理测试集合。


@pytest.fixture()  # 注册供用例复用的测试夹具。
def app_client(settings, store, tmp_path):  # 定义app_client的处理逻辑。
    registry = DocumentRegistry(tmp_path / "rag.db")  # 将登记表设为文档登记表的结果。
    embedder = HashEmbedder()  # 将向量服务设为摘要向量编码器的结果。
    lexical = LexicalIndex()  # 将关键词索引设为关键词索引的结果。
    retriever = HybridRetriever(  # 将检索器设为混合检索器的结果。
        settings.retrieval, store, embedder, lexical, KeywordReranker()  # 为KeywordReranker提供输入。
    )  # 结束当前多行表达式。
    generator = EchoGenerator()  # 将生成器设为EchoGenerator的结果。
    answerer = Answerer(settings, generator)  # 将问答服务设为问答服务的结果。
    pipeline = IngestionPipeline(  # 将入库流程设为入库流程的结果。
        settings, store, registry, embedder, Chunker(settings.chunking, embedder=embedder)  # 为分块器提供输入。
    )  # 结束当前多行表达式。
    services = Services(  # 将服务组件设为应用服务集合的结果。
        settings=settings,  # 传入配置参数。
        store=store,  # 传入向量库参数。
        registry=registry,  # 传入登记表参数。
        embedder=embedder,  # 传入向量服务参数。
        reranker=KeywordReranker(),  # 传入重排器参数。
        generator=generator,  # 传入生成器参数。
        lexical=lexical,  # 传入关键词索引参数。
        retriever=retriever,  # 传入检索器参数。
        answerer=answerer,  # 传入问答服务参数。
        pipeline=pipeline,  # 传入入库流程参数。
        dimension=DIM,  # 传入向量维度参数。
    )  # 结束当前多行表达式。
    client = TestClient(create_app(settings, services))  # 将客户端设为测试客户端的结果。
    client.services = services  # 保存传入的服务组件。
    client.registry = registry  # 保存传入的登记表。
    yield client  # 向调用方逐项提供客户端。
    store.client.delete(store.child_name, filter='doc_id != ""')  # 向子块集合名发送删除请求。
    store.client.delete(store.parent_name, filter='doc_id != ""')  # 向父块集合名发送删除请求。


def _events(response) -> list[tuple[str, dict]]:  # 将流式响应解析为事件名与载荷。
    events = []  # 初始化事件集合为空列表。
    for block in response.text.split("\n\n"):  # 遍历切分的结果，逐项取得block。
        if not block.strip():  # 当不满足“清理首尾空白后的block”时进入此分支。
            continue  # 跳过本项，继续处理下一项。
        event = None  # 将事件设为空值。
        data: list[str] = []  # 初始化数据为空列表。
        for line in block.split("\n"):  # 遍历切分的结果，逐项取得文本行。
            if line.startswith("event: "):  # 当前缀匹配的结果时进入此分支。
                event = line[7:]  # 将事件设为文本行的指定区间。
            elif line.startswith("data: "):  # 当前缀匹配的结果时进入此分支。
                data.append(line[6:])  # 将文本行的指定区间加入数据。
        if event:  # 当事件有效时进入此分支。
            events.append((event, json.loads("\n".join(data)) if data else {}))  # 将组合值加入事件集合。
    return events  # 返回事件集合。


MARKDOWN_DOC = {  # 将MARKDOWN_DOC设为字段映射。
    "name": "guide.md",  # 填入名称字段。
    # 保留多行文本中的换行和缩进。
    "body": """# 检索系统指南

## 混合检索原理

混合检索把稀疏召回与稠密召回结合。稀疏召回基于 BM25 做关键词匹配，
稠密召回基于向量做语义匹配。两路结果用 RRF 融合，避免单一分数尺度的问题。

## 父子文档

父子文档让检索与生成使用不同粒度。子块用于向量检索，父块交给模型阅读。
这样既能精确命中，又能提供完整上下文。

## 重排环节

重排对融合后的候选重新打分。重排的输入包含标题路径，
因此脱离标题的片段也能被正确理解。
""",
}  # 结束本组字段。


class TestFullChain:  # 定义 TestFullChain 类型。
    def test_upload_then_ask_finds_the_new_document(self, app_client):  # 验证上传完成后立刻提问可以命中新文档。
        """验证上传完成后立刻提问可以命中新文档。"""
        upload = app_client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs",  # 向文档管理接口提交本次测试请求。
            files={"files": (MARKDOWN_DOC["name"], MARKDOWN_DOC["body"].encode(), "text/markdown")},  # 传入文件列表参数。
        )  # 结束当前多行表达式。
        names = [name for name, _ in _events(upload)]  # 将名称列表设为从解析事件序列的结果中提取的各项。
        assert "done" in names  # 确认“done”包含在名称列表。

        chat = app_client.post("/api/chat", json={"question": "混合检索是怎么工作的？"})  # 向“/api/chat”发送提交请求并保存响应。
        events = _events(chat)  # 将事件集合设为解析事件序列的结果。

        sources = [payload for name, payload in events if name == "sources"][0]["sources"]  # 将来源列表设为从事件集合中提取的各项中对应的项中的来源列表。
        assert sources  # 确认来源列表有效。
        assert sources[0]["source"] == "guide.md"  # 确认来源列表的首项中的来源等于“guide.md”。

    def test_answer_streams_and_cites_sources(self, app_client):  # 验证回答逐段到达且带有引用。
        app_client.post(  # 向“/api/docs”发送提交请求。
            "/api/docs",  # 向文档管理接口提交本次测试请求。
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},  # 传入文件列表参数。
        )  # 结束当前多行表达式。

        events = _events(app_client.post("/api/chat", json={"question": "父子文档有什么用？"}))  # 将事件集合设为解析事件序列的结果。

        tokens = [payload["text"] for name, payload in events if name == "token"]  # 将词元列表设为从事件集合中提取的各项。
        assert "".join(tokens).startswith("根据")  # 确认前缀匹配的结果。
        done = [payload for name, payload in events if name == "done"][0]  # 将完成状态设为从事件集合中提取的各项中对应的项。
        assert done["answer"]  # 确认完成状态中的回答。
        assert not done["truncated"]  # 确认不满足“完成状态中的truncated”。

    def test_contexts_reach_the_model(self, app_client):  # 验证检索上下文实际传入生成模型。
        app_client.post(  # 向“/api/docs”发送提交请求。
            "/api/docs",  # 向文档管理接口提交本次测试请求。
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},  # 传入文件列表参数。
        )  # 结束当前多行表达式。

        app_client.post("/api/chat", json={"question": "重排为什么需要标题路径？"})  # 向“/api/chat”发送提交请求。

        prompts = app_client.services.answerer.generator.prompts  # 取出当前使用的提示消息记录。
        assert prompts  # 确认提示消息记录有效。
        # 模型应读到扩展后的父块上下文。
        assert "父块" in prompts[0] or "重排" in prompts[0]  # 确认“父块”包含在提示消息记录中对应的项或“重排”包含在提示消息记录中对应的项。


class TestConsistency:  # 定义 TestConsistency 类型。
    def test_document_is_retrievable_immediately_after_upload(self, app_client):  # 从上传到检索验证强一致性的即时可见效果。
        """从上传到检索验证强一致性的即时可见效果。"""
        body = "# 即时检索\n\n上传后立刻提问应该能命中这段内容。\n"  # 设置正文。

        app_client.post("/api/docs", files={"files": ("fresh.md", body.encode(), "text/markdown")})  # 向“/api/docs”发送提交请求。
        events = _events(app_client.post("/api/chat", json={"question": "上传后立刻提问"}))  # 将事件集合设为解析事件序列的结果。

        sources = [payload for name, payload in events if name == "sources"][0]["sources"]  # 将来源列表设为从事件集合中提取的各项中对应的项中的来源列表。
        assert any(s["source"] == "fresh.md" for s in sources)  # 确认至少一项满足“片段中的来源等于“fresh.md””。

    def test_removed_document_disappears_from_both_channels(self, app_client):  # 验证删除文档后两路召回均不再命中。
        body = "# 将被删除\n\n这段内容随后会被删除，不应再被检索到。\n"  # 设置正文。
        app_client.post("/api/docs", files={"files": ("temp.md", body.encode(), "text/markdown")})  # 向“/api/docs”发送提交请求。

        docs = app_client.get("/api/docs").json()["documents"]  # 将docs设为解析后的响应数据中的文档。
        doc_id = next(d["doc_id"] for d in docs if d["name"] == "temp.md")  # 将文档标识设为next的结果。
        app_client.delete(f"/api/docs/{doc_id}")  # 向格式化文本发送删除请求。

        events = _events(app_client.post("/api/chat", json={"question": "这段内容随后会被删除"}))  # 将事件集合设为解析事件序列的结果。
        sources = [payload for name, payload in events if name == "sources"][0]["sources"]  # 将来源列表设为从事件集合中提取的各项中对应的项中的来源列表。
        assert all(s["source"] != "temp.md" for s in sources)  # 确认每一项均满足“片段中的来源不等于“temp.md””。

        names = [d["name"] for d in app_client.get("/api/docs").json()["documents"]]  # 将名称列表设为从解析后的响应数据中的文档中提取的各项。
        assert "temp.md" not in names  # 确认“temp.md”未包含在名称列表。

    def test_reupload_replaces_rather_than_duplicates(self, app_client):  # 验证重复上传替换旧版本。
        first = "# 版本一\n\n旧版本的内容在这里。\n"  # 将首项设为文本内容。
        app_client.post("/api/docs", files={"files": ("same.md", first.encode(), "text/markdown")})  # 向“/api/docs”发送提交请求。

        second = "# 版本二\n\n新版本的内容替换了旧版本。\n"  # 将第二项设为文本内容。
        app_client.post("/api/docs", files={"files": ("same.md", second.encode(), "text/markdown")})  # 向“/api/docs”发送提交请求。

        docs = [d for d in app_client.get("/api/docs").json()["documents"] if d["name"] == "same.md"]  # 将docs设为从解析后的响应数据中的文档中提取的各项。
        assert len(docs) == 1  # 确认docs的数量等于1。
        assert docs[0]["state"] == "active"  # 确认docs的首项中的状态等于“active”。
        assert docs[0]["parent_count"] >= 1  # 确认docs的首项中的父块数不小于1。


class TestFailurePaths:  # 定义 TestFailurePaths 类型。
    def test_unsupported_type_is_rejected_in_the_stream(self, app_client):  # 验证事件流报告不支持的文件格式。
        response = app_client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs", files={"files": ("notes.txt", b"plain text", "text/plain")}  # 填入文件列表字段。
        )  # 结束当前多行表达式。
        events = _events(response)  # 将事件集合设为解析事件序列的结果。

        # 上传或解析阶段均可拒绝无效文件，但客户端应收到原因，索引应保持为空。
        failed = [payload for name, payload in events if name in {"failed", "rejected"}]  # 将失败设为从事件集合中提取的各项。
        assert failed  # 确认失败有效。
        assert app_client.registry.list_usable() == []  # 确认列出有效文档的结果等于空列表。

    def test_scanned_pdf_reports_the_reason(self, app_client):  # 验证扫描件遭拒时显示原因。
        import pymupdf  # 引入版面解析库。

        pdf = pymupdf.open()  # 将便携文档设为打开文档的结果。
        pdf.new_page()  # 创建页面。
        content = pdf.tobytes()  # 将内容设为导出文档字节串的结果。
        pdf.close()  # 关闭便携文档并释放资源。

        response = app_client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs", files={"files": ("scan.pdf", content, "application/pdf")}  # 填入文件列表字段。
        )  # 结束当前多行表达式。
        failed = [payload for name, payload in _events(response) if name == "failed"]  # 将失败设为从解析事件序列的结果中提取的各项。

        assert failed  # 确认失败有效。
        assert failed[0]["stage"] == "parsing"  # 确认失败的首项中的阶段等于“parsing”。
        assert "scan" in failed[0]["error"].lower() or "image" in failed[0]["error"].lower()  # 确认“scan”包含在失败中对应的项中的错误的小写形式或“image”包含在失败中对应的项中的错误的小写形式。

    def test_question_with_no_documents_is_answered_honestly(self, app_client):  # 验证空库提问时明确告知缺少资料。
        events = _events(app_client.post("/api/chat", json={"question": "anything?"}))  # 将事件集合设为解析事件序列的结果。

        done = [payload for name, payload in events if name == "done"][0]  # 将完成状态设为从事件集合中提取的各项中对应的项。
        assert done["empty"] is True  # 确认完成状态中的为空为真。
        # 确认没有调用模型。
        assert app_client.services.answerer.generator.prompts == []  # 确认提示消息记录等于空列表。


class TestMultiTurn:  # 定义 TestMultiTurn 类型。
    def test_follow_up_uses_a_rewritten_query(self, app_client):  # 验证追问使用改写的问题检索。
        app_client.post(  # 向“/api/docs”发送提交请求。
            "/api/docs",  # 向文档管理接口提交本次测试请求。
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},  # 传入文件列表参数。
        )  # 结束当前多行表达式。
        history = [{"question": "混合检索是什么", "answer": "一种结合稀疏与稠密召回的方法"}]  # 将对话历史设为列表。

        events = _events(  # 将事件集合设为解析事件序列的结果。
            app_client.post(  # 为提交提供输入。
                "/api/chat", json={"question": "它有什么好处？", "history": history}  # 填入对话历史字段。
            )  # 结束当前多行表达式。
        )  # 结束当前多行表达式。

        rewritten = [payload for name, payload in events if name == "rewritten"]  # 将改写标记设为从事件集合中提取的各项。
        assert rewritten  # 确认改写标记有效。
        assert rewritten[0]["query"] == "改写后的问题"  # 确认改写标记的首项中的查询等于“改写后的问题”。

    def test_first_turn_is_not_rewritten(self, app_client):  # 验证首轮提问无需改写。
        app_client.post(  # 向“/api/docs”发送提交请求。
            "/api/docs",  # 向文档管理接口提交本次测试请求。
            files={"files": ("guide.md", MARKDOWN_DOC["body"].encode(), "text/markdown")},  # 传入文件列表参数。
        )  # 结束当前多行表达式。

        events = _events(app_client.post("/api/chat", json={"question": "混合检索是什么？"}))  # 将事件集合设为解析事件序列的结果。

        assert not [payload for name, payload in events if name == "rewritten"]  # 确认不满足“从事件集合中提取的各项”。

def test_restart_protects_existing_upload_when_model_changes(settings, tmp_path, monkeypatch):
    from rag_demo.api import services as service_module

    isolated = settings.model_copy(deep=True)
    suffix = uuid.uuid4().hex
    isolated.milvus.child_collection = f"schema_child_{suffix}"
    isolated.milvus.parent_collection = f"schema_parent_{suffix}"
    isolated.milvus.meta_collection = f"schema_meta_{suffix}"
    isolated.storage.sqlite_path = str(tmp_path / "schema.db")
    monkeypatch.setattr(service_module, "Embedder", lambda _: HashEmbedder())
    store = MilvusStore(isolated)

    def restart():
        graph = service_module.build_services(isolated)
        service_module.initialise(graph, verify=False)
        return TestClient(create_app(isolated, graph))

    def upload(client, body):
        return dict(_events(client.post("/api/docs", files={"files": ("same.md", body, "text/markdown")})))

    def snapshot(client):
        children = store.all_active_children()
        return (client.get("/api/docs").json(), store.read_meta(), children,
                store.fetch_parents(row["parent_id"] for row in children), store.collection_stats())

    original_model = isolated.embedding.model
    try:
        fresh = restart()
        assert "done" in upload(fresh, b"# Original\n\nFirst document.")
        before = snapshot(fresh)
        assert before[1] == {"embedding_model": original_model, "dimension": DIM}
        assert before[2] and before[3]
        assert snapshot(restart()) == before

        isolated.embedding.model = "different-model-at-same-dimension"
        incompatible = restart()
        report = incompatible.get("/api/health").json()
        assert any(d["name"] == "store_schema" and d["status"] == "misconfigured"
                   for d in report["dependencies"])
        assert original_model in upload(incompatible, b"# Replacement")["failed"]["error"]
        assert snapshot(incompatible) == before

        isolated.embedding.model = original_model
        assert "done" in upload(restart(), b"# Replacement\n\nSecond document.")
        assert store.read_meta() == before[1]
        assert store.all_active_children() != before[2]
    finally:
        store.drop_all()


def test_bm25_failure_and_zero_hits_have_distinct_chat_results(app_client, monkeypatch):
    from unittest.mock import Mock

    services = app_client.services
    upload = app_client.post("/api/docs", files={"files": ("guide.md", MARKDOWN_DOC["body"].encode())})
    assert "done" in dict(_events(upload))
    failing_search = Mock(side_effect=RuntimeError("BM25 failed"))
    monkeypatch.setattr(services.lexical._retriever, "retrieve", failing_search)
    failed = _events(app_client.post("/api/chat", json={"question": "混合检索"}))
    assert failed[-1][0] == "error"
    assert "BM25 failed" in failed[-1][1]["message"]
    assert services.answerer.generator.prompts == []

    # 停用词没有关键词命中，但仍能用向量候选生成回答。
    answered = dict(_events(app_client.post("/api/chat", json={"question": "the", "debug": True})))
    assert answered["debug"]["dense"] and not answered["debug"]["sparse"]
    assert "done" in answered and len(services.answerer.generator.prompts) == 1
    failing_search.assert_called_once()
