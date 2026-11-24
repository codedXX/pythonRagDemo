"""注入模拟服务来验证接口契约，包括事件顺序、错误映射及响应中的凭据保护。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。
from dataclasses import dataclass, field  # 从 dataclasses 引入本模块使用的类型和工具。

import pytest  # 引入测试框架。
from fastapi.testclient import TestClient  # 从 fastapi.testclient 导入测试客户端。

from rag_demo.api.app import create_app  # 从 rag_demo.api.app 导入创建应用。
from rag_demo.api.services import Services  # 从 rag_demo.api.services 导入应用服务集合。
from rag_demo.config import Settings  # 从 rag_demo.config 导入项目配置。
from rag_demo.retrieval.hybrid import Context, RetrievalOutcome  # 从 rag_demo.retrieval.hybrid 引入本模块使用的类型和工具。
from rag_demo.retrieval.lexical import LexicalIndex  # 从 rag_demo.retrieval.lexical 导入关键词索引。
from rag_demo.selfcheck import DependencyReport, SelfCheckReport, Status  # 从 rag_demo.selfcheck 引入本模块使用的类型和工具。
from rag_demo.store.registry import DocumentRecord, DocumentState  # 从 rag_demo.store.registry 引入本模块使用的类型和工具。


class FakeRegistry:  # 定义 FakeRegistry 类型。
    def __init__(self) -> None:  # 保存实例所需的配置和依赖。
        self.records: dict[str, DocumentRecord] = {}  # 初始化登记记录列表为空字典。
        self.removed: list[str] = []  # 初始化已删除项为空列表。

    def list_documents(self):  # 定义列出登记文档的处理逻辑。
        return list(self.records.values())  # 返回值列表的结果的列表形式。

    def list_usable(self):  # 定义列出有效文档的处理逻辑。
        return [r for r in self.records.values() if r.state is DocumentState.ACTIVE]  # 返回从值列表的结果中提取的各项。

    def get(self, doc_id):  # 定义读取的处理逻辑。
        return self.records.get(doc_id)  # 返回登记记录列表中的对应值。

    def remove(self, doc_id):  # 定义移除的处理逻辑。
        self.records.pop(doc_id, None)  # 调用pop。
        self.removed.append(doc_id)  # 将文档标识加入已删除项。

    def add(self, doc_id: str, name: str, state=DocumentState.ACTIVE):  # 定义add的处理逻辑。
        self.records[doc_id] = DocumentRecord(  # 将登记记录列表中对应的项设为文档登记记录的结果。
            doc_id=doc_id, name=name, state=state, parent_count=2, child_count=4  # 传入子块数参数。
        )  # 结束当前多行表达式。


class FakeRetriever:  # 定义 FakeRetriever 类型。
    def __init__(self, contexts=None, error=None) -> None:  # 保存实例所需的配置和依赖。
        self.contexts = contexts if contexts is not None else [_default_context()]  # 将上下文列表设为按条件选取的值。
        self.error = error  # 保存传入的错误。
        self.queries: list[str] = []  # 初始化queries为空列表。

    def retrieve(self, query):  # 定义检索的处理逻辑。
        self.queries.append(query)  # 将查询加入queries。
        if self.error:  # 当错误有效时进入此分支。
            raise self.error  # 向上抛出当前异常。
        return RetrievalOutcome(query=query, contexts=self.contexts)  # 返回混合检索结果的结果。

    def rebuild_lexical_index(self):  # 定义重建关键词索引的处理逻辑。
        return len(self.contexts)  # 返回上下文列表的长度。


class FakeAnswerer:  # 定义 FakeAnswerer 类型。
    def __init__(self, pieces=("Hello", " world"), rewrite=None, failing=False) -> None:  # 保存实例所需的配置和依赖。
        self.pieces = pieces  # 保存传入的片段列表。
        self.rewrite_result = rewrite  # 将rewrite_result设为改写。
        self.failing = failing  # 保存传入的failing。
        self.questions: list[str] = []  # 初始化questions为空列表。

    def rewrite(self, question, conversation):  # 定义改写的处理逻辑。
        if self.rewrite_result:  # 当rewrite_result有效时进入此分支。
            return self.rewrite_result, True  # 返回组合值。
        return question, False  # 返回组合值。

    def stream_answer(self, question, contexts, conversation=None):  # 定义逐段生成回答的处理逻辑。
        self.questions.append(question)  # 将问题加入questions。
        if self.failing:  # 当failing有效时进入此分支。
            raise RuntimeError("generator exploded")  # 向上抛出运行时错误。
        yield from self.pieces  # 向调用方逐项提供片段列表。


class FakePipeline:  # 定义 FakePipeline 类型。
    def __init__(self, error=None) -> None:  # 保存实例所需的配置和依赖。
        self.error = error  # 保存传入的错误。
        self.ingested: list[tuple[str, bytes]] = []  # 初始化ingested为空列表。

    def ingest(self, name, content, on_progress=None, doc_id=None):  # 定义入库的处理逻辑。
        from rag_demo.ingestion import IngestResult, IngestionError, Progress, Stage  # 从 rag_demo.ingestion 引入本模块使用的类型和工具。

        self.ingested.append((name, content))  # 将名称、内容加入ingested。
        for stage in (Stage.PARSING, Stage.CHUNKING, Stage.EMBEDDING, Stage.PERSISTING):  # 遍历组合值，逐项取得阶段。
            if on_progress:  # 当进度回调有效时进入此分支。
                on_progress(Progress(stage=stage, message=f"{stage.value}...", current=1, total=2))  # 进度回调。
        if self.error:  # 当错误有效时进入此分支。
            raise self.error  # 向上抛出当前异常。
        return IngestResult(  # 返回入库结果的结果。
            doc_id="d1", name=name, parent_count=2, child_count=4, char_count=500, duration_seconds=0.4  # 传入耗时秒数参数。
        )  # 结束当前多行表达式。


class FakeStore:  # 定义 FakeStore 类型。
    def __init__(self) -> None:  # 保存实例所需的配置和依赖。
        self.deleted: list[str] = []  # 初始化deleted为空列表。

    def delete_document(self, doc_id):  # 定义删除文档的处理逻辑。
        self.deleted.append(doc_id)  # 将文档标识加入deleted。


def _default_context() -> Context:  # 定义默认测试上下文的处理逻辑。
    return Context(  # 返回引用上下文的结果。
        parent_id="p1",  # 传入所属父块标识参数。
        text="the retrieved passage",  # 传入文本参数。
        breadcrumb="Chapter One",  # 传入标题路径参数。
        page=4,  # 传入页码参数。
        source_name="paper.pdf",  # 传入来源文件名参数。
        matched_child="p1_c0",  # 传入命中的子块标识参数。
        matched_text="the matched fragment",  # 传入命中的子块正文参数。
        rerank_score=0.87,  # 传入重排分数参数。
    )  # 结束当前多行表达式。


@dataclass  # 根据字段声明生成数据类的初始化和比较方法。
class Fakes:  # 定义 Fakes 类型。
    registry: FakeRegistry = field(default_factory=FakeRegistry)  # 设置登记表字段的默认值。
    retriever: FakeRetriever = field(default_factory=FakeRetriever)  # 设置检索器字段的默认值。
    answerer: FakeAnswerer = field(default_factory=FakeAnswerer)  # 设置问答服务字段的默认值。
    pipeline: FakePipeline = field(default_factory=FakePipeline)  # 设置入库流程字段的默认值。
    store: FakeStore = field(default_factory=FakeStore)  # 设置向量库字段的默认值。
    lexical: LexicalIndex = field(default_factory=LexicalIndex)  # 设置关键词索引字段的默认值。


def _client(fakes: Fakes, health: SelfCheckReport | None = None) -> TestClient:  # 定义测试客户端的处理逻辑。
    settings = Settings()  # 将配置设为项目配置的结果。
    services = Services(  # 将服务组件设为应用服务集合的结果。
        settings=settings,  # 传入配置参数。
        # 传入向量库参数。
        store=fakes.store,  # type: ignore[arg-type]
        # 传入登记表参数。
        registry=fakes.registry,  # type: ignore[arg-type]
        # 传入向量服务参数。
        embedder=None,  # type: ignore[arg-type]
        # 传入重排器参数。
        reranker=None,  # type: ignore[arg-type]
        # 传入生成器参数。
        generator=None,  # type: ignore[arg-type]
        lexical=fakes.lexical,  # 传入关键词索引参数。
        # 传入检索器参数。
        retriever=fakes.retriever,  # type: ignore[arg-type]
        # 传入问答服务参数。
        answerer=fakes.answerer,  # type: ignore[arg-type]
        # 传入入库流程参数。
        pipeline=fakes.pipeline,  # type: ignore[arg-type]
        selfcheck=health,  # 传入自检参数。
        dimension=1024,  # 传入向量维度参数。
    )  # 结束当前多行表达式。
    app = create_app(settings, services)  # 将应用设为创建应用的结果。
    return TestClient(app)  # 返回测试客户端的结果。


def _events(response) -> list[tuple[str, dict]]:  # 将事件流拆成事件名和载荷，供断言逐条核对。
    """将事件流拆成事件名和载荷，供断言逐条核对。"""
    events: list[tuple[str, dict]] = []  # 初始化事件集合为空列表。
    for block in response.text.split("\n\n"):  # 遍历切分的结果，逐项取得block。
        if not block.strip():  # 当不满足“清理首尾空白后的block”时进入此分支。
            continue  # 跳过本项，继续处理下一项。
        event = None  # 将事件设为空值。
        data: list[str] = []  # 初始化数据为空列表。
        for line in block.split("\n"):  # 遍历切分的结果，逐项取得文本行。
            if line.startswith("event: "):  # 当前缀匹配的结果时进入此分支。
                event = line[len("event: ") :]  # 将事件设为文本行的指定区间。
            elif line.startswith("data: "):  # 当前缀匹配的结果时进入此分支。
                data.append(line[len("data: ") :])  # 将文本行的指定区间加入数据。
        if event:  # 当事件有效时进入此分支。
            payload = json.loads("\n".join(data)) if data else {}  # 将载荷设为按条件选取的值。
            events.append((event, payload))  # 将事件、载荷加入事件集合。
    return events  # 返回事件集合。


class TestHealth:  # 定义 TestHealth 类型。
    def test_reports_dependency_status(self):  # 验证健康接口报告各项依赖状态。
        health = SelfCheckReport(  # 将health设为依赖自检报告的结果。
            dependencies=[  # 传入依赖项列表参数。
                DependencyReport("embedding", Status.HEALTHY, "ok", {"dimension": 1024}),  # 为单项依赖报告提供输入。
                DependencyReport("rerank", Status.MISCONFIGURED, "no key"),  # 为单项依赖报告提供输入。
            ],  # 结束列表。
            embedding_dimension=1024,  # 传入实测向量维度参数。
        )  # 结束当前多行表达式。
        client = _client(Fakes(), health)  # 将客户端设为测试客户端的结果。

        body = client.get("/api/health").json()  # 将正文设为解析后的响应数据。

        assert body["healthy"] is False  # 确认正文中的依赖健康状态为假。
        assert body["dimension"] == 1024  # 确认正文中的向量维度等于1024。
        by_name = {d["name"]: d["status"] for d in body["dependencies"]}  # 将by_name设为从正文中的依赖项列表中提取的映射。
        assert by_name["embedding"] == "healthy"  # 确认by_name中的向量等于“healthy”。
        assert by_name["rerank"] == "misconfigured"  # 确认by_name中的重排等于“misconfigured”。

    def test_never_exposes_secrets(self):  # 验证健康响应隐藏凭据。
        settings = Settings()  # 将配置设为项目配置的结果。
        settings.embedding.api_key = "sk-super-secret-value"  # 将接口密钥设为“sk-super-secret-value”。
        settings.llm.api_key = "sk-relay-secret-value"  # 将接口密钥设为“sk-relay-secret-value”。
        settings.rerank.api_key = "sk-rerank-secret-value"  # 将接口密钥设为“sk-rerank-secret-value”。
        fakes = Fakes()  # 将模拟服务设为模拟组件的结果。
        services = Services(  # 将服务组件设为应用服务集合的结果。
            settings=settings,  # 传入配置参数。
            store=fakes.store,  # 传入向量库参数。
            registry=fakes.registry,  # 传入登记表参数。
            embedder=None,  # 传入向量服务参数。
            reranker=None,  # 传入重排器参数。
            generator=None,  # 传入生成器参数。
            lexical=fakes.lexical,  # 传入关键词索引参数。
            retriever=fakes.retriever,  # 传入检索器参数。
            answerer=fakes.answerer,  # 传入问答服务参数。
            pipeline=fakes.pipeline,  # 传入入库流程参数。
            dimension=1024,  # 传入向量维度参数。
        )  # 结束当前多行表达式。

        response = create_app(settings, services)  # 将响应设为创建应用的结果。
        body = TestClient(response).get("/api/health").text  # 将正文设为文本。

        assert "sk-super-secret-value" not in body  # 确认“sk-super-secret-value”未包含在正文。
        assert "sk-relay-secret-value" not in body  # 确认“sk-relay-secret-value”未包含在正文。
        assert "sk-rerank-secret-value" not in body  # 确认“sk-rerank-secret-value”未包含在正文。


class TestChatStream:  # 定义 TestChatStream 类型。
    def test_sources_arrive_before_any_token(self):  # 检索来源应先于回答文本到达。
        """检索来源应先于回答文本到达。"""
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。

        names = [name for name, _ in events]  # 将名称列表设为从事件集合中提取的各项。
        assert "sources" in names  # 确认“sources”包含在名称列表。
        assert names.index("sources") < names.index("token")  # 确认索引的结果小于索引的结果。

    def test_tokens_are_streamed_individually(self):  # 验证回答片段分别推送。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。

        tokens = [payload["text"] for name, payload in events if name == "token"]  # 将词元列表设为从事件集合中提取的各项。
        assert tokens == ["Hello", " world"]  # 确认词元列表等于列表。

    def test_done_carries_the_full_answer(self):  # 验证结束事件携带完整回答。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。

        done = [payload for name, payload in events if name == "done"][0]  # 将完成状态设为从事件集合中提取的各项中对应的项。
        assert done["answer"] == "Hello world"  # 确认完成状态中的回答等于“Hello world”。
        assert done["truncated"] is False  # 确认完成状态中的截断标记为假。

    def test_sources_include_provenance(self):  # 验证每条来源包含定位信息。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。
        sources = [payload for name, payload in events if name == "sources"][0]["sources"]  # 将来源列表设为从事件集合中提取的各项中对应的项中的来源列表。

        assert sources[0]["source"] == "paper.pdf"  # 确认来源列表的首项中的来源等于“paper.pdf”。
        assert sources[0]["page"] == 4  # 确认来源列表的首项中的页码等于4。
        assert sources[0]["breadcrumb"] == "Chapter One"  # 确认来源列表的首项中的标题路径等于“Chapter One”。
        assert sources[0]["score"] == pytest.approx(0.87)  # 确认来源列表的首项中的分数等于approx的结果。

    def test_status_events_bracket_the_phases(self):  # 验证状态事件标识对应处理阶段。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。
        stages = [payload["stage"] for name, payload in events if name == "status"]  # 将阶段列表设为从事件集合中提取的各项。

        assert stages == ["rewriting", "retrieving"]  # 确认阶段列表等于列表。

    def test_rewritten_query_is_reported(self):  # 验证客户端能收到改写后的检索问题。
        fakes = Fakes(answerer=FakeAnswerer(rewrite="a standalone question"))  # 将模拟服务设为模拟组件的结果。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "it?"}))  # 将事件集合设为解析事件序列的结果。

        rewritten = [payload for name, payload in events if name == "rewritten"]  # 将改写标记设为从事件集合中提取的各项。
        assert rewritten[0]["query"] == "a standalone question"  # 确认改写标记的首项中的查询等于“a standalone question”。
        assert fakes.retriever.queries == ["a standalone question"]  # 确认fakes.retriever.queries等于列表。

    def test_empty_knowledge_base_is_explained(self):  # 验证知识库为空时返回明确原因。
        fakes = Fakes(retriever=FakeRetriever(contexts=[]))  # 将模拟服务设为模拟组件的结果。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "anything?"}))  # 将事件集合设为解析事件序列的结果。

        done = [payload for name, payload in events if name == "done"][0]  # 将完成状态设为从事件集合中提取的各项中对应的项。
        assert done["empty"] is True  # 确认完成状态中的为空为真。
        # 此时应跳过生成调用。
        assert fakes.answerer.questions == []  # 确认fakes.answerer.questions等于空列表。

    def test_mid_stream_failure_is_reported_as_truncated(self):  # 验证流中途失败时标记回答截断。
        fakes = Fakes(answerer=FakeAnswerer(failing=True))  # 将模拟服务设为模拟组件的结果。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。

        errors = [payload for name, payload in events if name == "error"]  # 将errors设为从事件集合中提取的各项。
        assert errors and errors[0]["truncated"] is True  # 确认errors且errors中对应的项中的truncated为真。
        done = [payload for name, payload in events if name == "done"][0]  # 将完成状态设为从事件集合中提取的各项中对应的项。
        assert done["truncated"] is True  # 确认完成状态中的截断标记为真。

    def test_retrieval_failure_is_reported_without_tokens(self):  # 验证检索失败时报告错误并跳过生成。
        fakes = Fakes(retriever=FakeRetriever(error=RuntimeError("dense channel down")))  # 将模拟服务设为模拟组件的结果。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "what?"}))  # 将事件集合设为解析事件序列的结果。

        names = [name for name, _ in events]  # 将名称列表设为从事件集合中提取的各项。
        assert "error" in names  # 确认“error”包含在名称列表。
        assert "token" not in names  # 确认“token”未包含在名称列表。

    def test_missing_question_is_rejected(self):  # 验证缺少问题的请求被拒绝。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        response = client.post("/api/chat", json={})  # 向“/api/chat”发送提交请求并保存响应。

        assert response.status_code == 400  # 确认响应状态码等于400。

    def test_history_is_accepted(self):  # 验证接口接收客户端传来的历史。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        response = client.post(  # 向“/api/chat”发送提交请求并保存响应。
            "/api/chat",  # 向问答接口发送本轮请求。
            json={  # 传入结构化数据参数。
                "question": "follow up",  # 填入问题字段。
                "history": [{"question": "first", "answer": "first answer"}],  # 填入回答字段。
            },  # 结束本组字段。
        )  # 结束当前多行表达式。

        assert response.status_code == 200  # 确认响应状态码等于200。

    def test_debug_event_is_opt_in(self):  # 验证只有开启调试时才推送调试事件。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        without = _events(client.post("/api/chat", json={"question": "q"}))  # 将without设为解析事件序列的结果。
        with_debug = _events(client.post("/api/chat", json={"question": "q", "debug": True}))  # 将with_debug设为解析事件序列的结果。

        assert "debug" not in [name for name, _ in without]  # 确认“debug”未包含在从without中提取的各项。
        assert "debug" in [name for name, _ in with_debug]  # 确认“debug”包含在从with_debug中提取的各项。

    def test_debug_reports_every_stage(self):  # 验证调试结果覆盖每个检索阶段。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        events = _events(client.post("/api/chat", json={"question": "q", "debug": True}))  # 将事件集合设为解析事件序列的结果。
        debug = [payload for name, payload in events if name == "debug"][0]  # 将调试开关设为从事件集合中提取的各项中对应的项。

        assert set(debug) >= {"query", "dense", "sparse", "fused", "reranked"}  # 确认去重集合至少为集合。


class TestUploadStream:  # 定义 TestUploadStream 类型。
    def test_progress_events_precede_completion(self):  # 验证进度事件先于完成通知。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        response = client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs", files={"files": ("notes.md", b"# hi\n\ntext", "text/markdown")}  # 填入文件列表字段。
        )  # 结束当前多行表达式。
        names = [name for name, _ in _events(response)]  # 将名称列表设为从解析事件序列的结果中提取的各项。

        assert names[0] == "start"  # 确认名称列表的首项等于“start”。
        assert "progress" in names  # 确认“progress”包含在名称列表。
        assert names.index("progress") < names.index("done")  # 确认索引的结果小于索引的结果。
        assert names[-1] == "complete"  # 确认名称列表中对应的项等于“complete”。

    def test_stages_are_reported_in_order(self):  # 验证按处理顺序报告各阶段。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        response = client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs", files={"files": ("notes.md", b"content", "text/markdown")}  # 填入文件列表字段。
        )  # 结束当前多行表达式。
        stages = [  # 将阶段列表设为从解析事件序列的结果中提取的各项。
            payload["stage"]  # 组合当前结果所需的各项数据。
            for name, payload in _events(response)  # 逐项处理推导式中的name, payload。
            if name == "progress"  # 按条件筛选或选择当前值。
        ]  # 结束列表。

        assert stages == ["parsing", "chunking", "embedding", "persisting"]  # 确认阶段列表等于列表。

    def test_failure_names_the_stage(self):  # 验证错误信息包含失败阶段。
        from rag_demo.ingestion import IngestionError, Stage  # 从 rag_demo.ingestion 引入本模块使用的类型和工具。

        fakes = Fakes(pipeline=FakePipeline(error=IngestionError(Stage.PARSING, "scanned pdf")))  # 将模拟服务设为模拟组件的结果。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        response = client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs", files={"files": ("scan.pdf", b"%PDF-1.4", "application/pdf")}  # 填入文件列表字段。
        )  # 结束当前多行表达式。
        failed = [payload for name, payload in _events(response) if name == "failed"]  # 将失败设为从解析事件序列的结果中提取的各项。

        assert failed[0]["stage"] == "parsing"  # 确认失败的首项中的阶段等于“parsing”。
        assert "scanned" in failed[0]["error"]  # 确认失败的首项中的错误包含“scanned”。

    def test_oversized_file_is_rejected_before_ingestion(self):  # 验证超限文件在进入入库流程前被拒绝。
        fakes = Fakes()  # 将模拟服务设为模拟组件的结果。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。
        # 配置按兆字节计量；临时替换字节上限，以小文件覆盖超限路径。
        client.app.state.settings.storage.max_upload_mb = 0  # 将单文件上传上限（兆字节）设为0。

        response = client.post(  # 向“/api/docs”发送提交请求并保存响应。
            "/api/docs", files={"files": ("big.md", b"x" * 10, "text/markdown")}  # 填入文件列表字段。
        )  # 结束当前多行表达式。
        names = [name for name, _ in _events(response)]  # 将名称列表设为从解析事件序列的结果中提取的各项。

        assert "rejected" in names  # 确认“rejected”包含在名称列表。
        assert fakes.pipeline.ingested == []  # 确认fakes.pipeline.ingested等于空列表。


class TestDocumentManagement:  # 定义 TestDocumentManagement 类型。
    def test_listing_returns_documents_with_state(self):  # 验证文档列表包含处理状态。
        fakes = Fakes()  # 将模拟服务设为模拟组件的结果。
        fakes.registry.add("d1", "paper.pdf")  # 将“d1”加入登记表。
        fakes.registry.add("d2", "broken.pdf", state=DocumentState.FAILED)  # 将“d2”加入登记表。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        body = client.get("/api/docs").json()  # 将正文设为解析后的响应数据。

        states = {d["name"]: d["state"] for d in body["documents"]}  # 将states设为从正文中的文档中提取的映射。
        assert states == {"paper.pdf": "active", "broken.pdf": "failed"}  # 确认states等于字段映射。

    def test_deleting_removes_content_and_returns_id(self):  # 验证删除成功后返回标识并清除文档。
        fakes = Fakes()  # 将模拟服务设为模拟组件的结果。
        fakes.registry.add("d1", "paper.pdf")  # 将“d1”加入登记表。
        client = _client(fakes)  # 将客户端设为测试客户端的结果。

        response = client.delete("/api/docs/d1")  # 向“/api/docs/d1”发送删除请求并保存响应。

        assert response.json()["removed"] == "d1"  # 确认解析后的响应数据中的已删除项等于“d1”。
        assert fakes.store.deleted == ["d1"]  # 确认fakes.store.deleted等于列表。

    def test_deleting_unknown_document_is_404(self):  # 验证删除未知标识时返回未找到状态。
        client = _client(Fakes())  # 将客户端设为测试客户端的结果。

        assert client.delete("/api/docs/nope").status_code == 404  # 确认响应状态码等于404。


def test_index_page_is_served(monkeypatch, tmp_path):  # 页面由同一进程从静态目录提供。
    """页面由同一进程从静态目录提供。"""
    static = tmp_path / "static"  # 将静态资源设为临时目录下的“static”。
    static.mkdir()  # 按需创建静态资源。
    (static / "index.html").write_text("<html><body>rag</body></html>", encoding="utf-8")  # 将内容写入静态资源下的“index.html”。
    monkeypatch.setattr("rag_demo.api.app.STATIC_DIR", static)  # 临时替换文本内容的静态资源以控制测试依赖。

    client = _client(Fakes())  # 将客户端设为测试客户端的结果。

    response = client.get("/")  # 向“/”发送读取请求并保存响应。

    assert response.status_code == 200  # 确认响应状态码等于200。
    assert "rag" in response.text  # 确认“rag”包含在文本。


def test_index_contains_no_secrets(monkeypatch, tmp_path):  # 验证首页不含服务端密钥。
    static = tmp_path / "static"  # 将静态资源设为临时目录下的“static”。
    static.mkdir()  # 按需创建静态资源。
    (static / "index.html").write_text("<html><body>page</body></html>", encoding="utf-8")  # 将内容写入静态资源下的“index.html”。
    monkeypatch.setattr("rag_demo.api.app.STATIC_DIR", static)  # 临时替换文本内容的静态资源以控制测试依赖。
    settings = Settings()  # 将配置设为项目配置的结果。
    settings.llm.api_key = "sk-must-not-leak"  # 将接口密钥设为“sk-must-not-leak”。
    fakes = Fakes()  # 将模拟服务设为模拟组件的结果。
    services = Services(  # 将服务组件设为应用服务集合的结果。
        settings=settings,  # 传入配置参数。
        store=fakes.store,  # 传入向量库参数。
        registry=fakes.registry,  # 传入登记表参数。
        embedder=None,  # 传入向量服务参数。
        reranker=None,  # 传入重排器参数。
        generator=None,  # 传入生成器参数。
        lexical=fakes.lexical,  # 传入关键词索引参数。
        retriever=fakes.retriever,  # 传入检索器参数。
        answerer=fakes.answerer,  # 传入问答服务参数。
        pipeline=fakes.pipeline,  # 传入入库流程参数。
        dimension=1024,  # 传入向量维度参数。
    )  # 结束当前多行表达式。
    client = TestClient(create_app(settings, services))  # 将客户端设为测试客户端的结果。

    body = client.get("/").text  # 将正文设为文本。

    assert "sk-must-not-leak" not in body  # 确认“sk-must-not-leak”未包含在正文。

def test_retrieval_error_is_not_described_as_a_generation_error():  # 两类故障需要不同的排查步骤，提示应体现故障来源。
    """两类故障需要不同的排查步骤，提示应体现故障来源。"""
    from rag_demo.api.routes import describe_retrieval_error  # 从 rag_demo.api.routes 导入整理检索错误提示。
    from rag_demo.errors import ConfigurationIssue  # 从 rag_demo.errors 导入供应商配置异常。

    message = describe_retrieval_error(ConfigurationIssue("embedding", "no API key"))  # 将消息设为整理检索错误提示的结果。

    assert "Retrieval" in message  # 确认“Retrieval”包含在消息。
    assert "Generation" not in message  # 确认“Generation”未包含在消息。
    assert "embedding" in message.lower() or "no API key" in message  # 确认“embedding”包含在消息的小写形式或“no API key”包含在消息。
