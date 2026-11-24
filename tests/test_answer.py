"""验证提示词组装、对话历史长度和追问改写。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pytest  # 引入测试框架。

from rag_demo.config import Settings  # 从 rag_demo.config 导入项目配置。
from rag_demo.errors import ConfigurationIssue  # 从 rag_demo.errors 导入供应商配置异常。
from rag_demo.generation.answer import (  # 从 rag_demo.generation.answer 引入本模块使用的类型和工具。
    NO_CONTEXT_ANSWER,  # 导入缺少上下文时的提示。
    Answerer,  # 导入问答服务。
    Conversation,  # 导入对话历史。
    Turn,  # 导入一轮问答。
    build_messages,  # 导入组装回答消息。
    describe_generation_error,  # 导入整理生成错误提示。
    format_contexts,  # 导入格式化引用段落。
    needs_rewrite,  # 导入判断是否需要改写。
)  # 结束当前多行表达式。
from rag_demo.retrieval.hybrid import Context  # 从 rag_demo.retrieval.hybrid 导入引用上下文。


def _context(  # 组装带有来源信息的测试上下文。
    text: str = "some passage",  # 接收文本参数。
    source: str = "paper.pdf",  # 接收来源参数。
    page: int = 3,  # 接收页码参数。
    breadcrumb: str = "Chapter One",  # 接收标题路径参数。
) -> Context:  # 结束参数声明，进入函数或语句主体。
    return Context(  # 返回引用上下文的结果。
        parent_id="p1",  # 传入所属父块标识参数。
        text=text,  # 传入文本参数。
        breadcrumb=breadcrumb,  # 传入标题路径参数。
        page=page,  # 传入页码参数。
        source_name=source,  # 传入来源文件名参数。
    )  # 结束当前多行表达式。


def _settings(**overrides) -> Settings:  # 在测试默认配置上应用本用例的覆盖项。
    settings = Settings()  # 将配置设为项目配置的结果。
    for key, value in overrides.items():  # 遍历覆盖项的键值对，逐项取得键、值。
        setattr(settings.llm, key, value)  # 临时替换生成模型的键以控制测试依赖。
    return settings  # 返回配置。


class FakeGenerator:  # 模拟生成服务并记录收到的请求。
    """模拟生成服务并记录收到的请求。"""

    def __init__(self, answer: str = "the answer", rewrite: str | None = None, failing: bool = False) -> None:  # 保存实例所需的配置和依赖。
        self.answer = answer  # 保存传入的回答。
        self.rewrite = rewrite  # 保存传入的改写。
        self.failing = failing  # 保存传入的failing。
        self.calls: list[list] = []  # 初始化调用记录为空列表。

    def stream_chat(self, messages):  # 定义流式生成回复的处理逻辑。
        self.calls.append(list(messages))  # 将消息列表的列表形式加入调用记录。
        if self.failing:  # 当failing有效时进入此分支。
            raise ConfigurationIssue("llm", "bad key")  # 向上抛出供应商配置异常。
        for piece in self.answer.split():  # 遍历切分的结果，逐项取得片段。
            yield piece + " "  # 向调用方逐项提供片段加上“ ”。

    def complete(self, messages):  # 定义获取完整回答的处理逻辑。
        self.calls.append(list(messages))  # 将消息列表的列表形式加入调用记录。
        if self.failing:  # 当failing有效时进入此分支。
            raise ConfigurationIssue("llm", "bad key")  # 向上抛出供应商配置异常。
        return self.rewrite if self.rewrite is not None else self.answer  # 返回按条件选取的值。


class TestContextFormatting:  # 定义 TestContextFormatting 类型。
    def test_contexts_are_numbered(self):  # 验证引用片段按顺序编号。
        rendered = format_contexts([_context("first"), _context("second")])  # 将渲染结果设为格式化引用段落的结果。

        assert "[1]" in rendered  # 确认“[1]”包含在渲染结果。
        assert "[2]" in rendered  # 确认“[2]”包含在渲染结果。

    def test_provenance_is_included(self):  # 验证上下文带有文档来源信息。
        rendered = format_contexts([_context("body", source="report.pdf", page=7)])  # 将渲染结果设为格式化引用段落的结果。

        assert "report.pdf" in rendered  # 确认“report.pdf”包含在渲染结果。
        assert "page 7" in rendered  # 确认“page 7”包含在渲染结果。
        assert "Chapter One" in rendered  # 确认“Chapter One”包含在渲染结果。

    def test_missing_page_is_omitted_cleanly(self):  # 验证页码缺失时省略该位置描述。
        rendered = format_contexts([_context("body", page=0)])  # 将渲染结果设为格式化引用段落的结果。

        assert "page 0" not in rendered  # 确认“page 0”未包含在渲染结果。
        assert "body" in rendered  # 确认“body”包含在渲染结果。

    def test_empty_breadcrumb_is_omitted(self):  # 验证标题路径为空时省略路径。
        rendered = format_contexts([_context("body", breadcrumb="")])  # 将渲染结果设为格式化引用段落的结果。

        assert ", \n" not in rendered  # 确认文本内容未包含在渲染结果。


class TestPromptAssembly:  # 定义 TestPromptAssembly 类型。
    def test_system_prompt_requires_grounding(self):  # 验证系统提示要求回答以资料为依据。
        messages = build_messages("q", [_context()])  # 将消息列表设为组装回答消息的结果。

        system = messages[0]  # 将系统设为消息列表中对应的项。
        assert system.role == "system"  # 确认角色等于“system”。
        assert "only from the numbered context" in system.content  # 确认文本内容包含在内容。
        assert "do not cover it" in system.content.lower()  # 确认“do not cover it”包含在内容的小写形式。

    def test_contexts_appear_before_the_question(self):  # 验证引用上下文排在提问前面。
        messages = build_messages("what is X", [_context("passage about X")])  # 将消息列表设为组装回答消息的结果。

        user = messages[-1]  # 将用户设为消息列表中对应的项。
        assert user.role == "user"  # 确认角色等于“user”。
        assert user.content.index("passage about X") < user.content.index("what is X")  # 确认索引的结果小于索引的结果。

    def test_history_precedes_the_new_question(self):  # 验证历史消息排在本轮问题前面。
        conversation = Conversation(turns=[Turn(question="first q", answer="first a")])  # 将对话历史设为对话历史的结果。

        messages = build_messages("second q", [_context()], conversation)  # 将消息列表设为组装回答消息的结果。

        contents = [m.content for m in messages]  # 将contents设为从消息列表中提取的各项。
        assert any("first q" in c for c in contents)  # 确认至少一项满足“候选项包含“first q””。
        assert contents[-1].endswith("second q")  # 确认endswith的结果。

    def test_no_history_means_no_history_messages(self):  # 验证没有历史时只组装本轮消息。
        messages = build_messages("q", [_context()])  # 将消息列表设为组装回答消息的结果。

        assert len(messages) == 2  # 确认消息列表的数量等于2。


class TestConversationBounds:  # 定义 TestConversationBounds 类型。
    def test_history_is_capped(self):  # 验证历史轮数受到上限约束。
        conversation = Conversation(max_turns=3)  # 将对话历史设为对话历史的结果。

        for index in range(10):  # 遍历指定的下标范围，逐项取得索引。
            conversation.append(f"q{index}", f"a{index}")  # 将格式化文本加入对话历史。

        assert len(conversation.turns) == 3  # 确认对话历史的历史轮次列表的数量等于3。
        # 达到轮数上限时应保留最近的对话。
        assert conversation.turns[-1].question == "q9"  # 确认问题等于“q9”。
        assert conversation.turns[0].question == "q7"  # 确认问题等于“q7”。

    def test_messages_alternate_roles(self):  # 验证用户和助手角色按轮次交替。
        conversation = Conversation()  # 将对话历史设为对话历史的结果。
        conversation.append("q", "a")  # 将“q”加入对话历史。

        roles = [m.role for m in conversation.as_messages()]  # 将roles设为从生成历史消息列表的结果中提取的各项。

        assert roles == ["user", "assistant"]  # 确认roles等于列表。


class TestRewriteDecision:  # 定义 TestRewriteDecision 类型。
    def test_first_turn_is_never_rewritten(self):  # 验证首轮问题直接用于检索。
        assert needs_rewrite("what is retrieval", Conversation()) is False  # 确认判断是否需要改写的结果为假。

    def test_anaphoric_follow_up_is_rewritten(self):  # 验证依赖前文指代的追问会被改写。
        conversation = Conversation(turns=[Turn("tell me about RRF", "RRF fuses ranks")])  # 将对话历史设为对话历史的结果。

        assert needs_rewrite("why does it work?", conversation) is True  # 确认判断是否需要改写的结果为真。

    def test_chinese_anaphora_is_rewritten(self):  # 验证中文指代词触发问题改写。
        conversation = Conversation(turns=[Turn("介绍一下混合检索", "混合检索...")])  # 将对话历史设为对话历史的结果。

        assert needs_rewrite("它的缺点是什么？", conversation) is True  # 确认判断是否需要改写的结果为真。

    def test_self_contained_question_is_not_rewritten(self):  # 验证信息完整的问题直接检索。
        conversation = Conversation(turns=[Turn("q", "a")])  # 将对话历史设为对话历史的结果。
        question = "How does reciprocal rank fusion combine two different ranked lists?"  # 设置问题。

        assert needs_rewrite(question, conversation) is False  # 确认判断是否需要改写的结果为假。

    def test_very_short_question_is_rewritten(self):  # 验证极短的追问会被改写。
        conversation = Conversation(turns=[Turn("q", "a")])  # 将对话历史设为对话历史的结果。

        assert needs_rewrite("more?", conversation) is True  # 确认判断是否需要改写的结果为真。


class TestAnswerer:  # 定义 TestAnswerer 类型。
    def test_answer_uses_contexts(self):  # 验证生成回答时传入检索上下文。
        generator = FakeGenerator(answer="grounded answer")  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。

        text = answerer.answer("q", [_context("the passage")])  # 将文本设为回答的结果。

        assert "grounded" in text  # 确认“grounded”包含在文本。
        assert any("the passage" in m.content for m in generator.calls[0])  # 确认至少一项满足“m的内容包含“the passage””。

    def test_empty_contexts_short_circuit_without_calling_model(self):  # 验证没有上下文时直接返回提示并跳过模型调用。
        generator = FakeGenerator()  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。

        text = answerer.answer("q", [])  # 将文本设为回答的结果。

        assert text == NO_CONTEXT_ANSWER  # 确认文本等于缺少上下文时的提示。
        assert generator.calls == []  # 确认调用记录等于空列表。

    def test_rewrite_is_skipped_for_first_turn(self):  # 验证生成器在首轮跳过改写。
        generator = FakeGenerator()  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。

        query, rewritten = answerer.rewrite("what is retrieval", Conversation())  # 将查询、改写标记设为改写的结果。

        assert rewritten is False  # 确认改写标记为假。
        assert query == "what is retrieval"  # 确认查询等于“what is retrieval”。
        assert generator.calls == []  # 确认调用记录等于空列表。

    def test_rewrite_is_used_for_follow_up(self):  # 验证后续追问使用改写结果。
        generator = FakeGenerator(rewrite="why does reciprocal rank fusion work")  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。
        conversation = Conversation(turns=[Turn("tell me about RRF", "RRF fuses ranks")])  # 将对话历史设为对话历史的结果。

        query, rewritten = answerer.rewrite("why does it work?", conversation)  # 将查询、改写标记设为改写的结果。

        assert rewritten is True  # 确认改写标记为真。
        assert query == "why does reciprocal rank fusion work"  # 确认查询等于文本内容。

    def test_rewrite_strips_quotes(self):  # 验证改写结果去除包裹它的引号。
        generator = FakeGenerator(rewrite='"a quoted rewrite"')  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。
        conversation = Conversation(turns=[Turn("q", "a")])  # 将对话历史设为对话历史的结果。

        query, _ = answerer.rewrite("it?", conversation)  # 将查询、_设为改写的结果。

        assert query == "a quoted rewrite"  # 确认查询等于“a quoted rewrite”。

    def test_failing_rewrite_falls_back_to_original(self):  # 验证改写失败后沿用原问题。
        generator = FakeGenerator(failing=True)  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。
        conversation = Conversation(turns=[Turn("q", "a")])  # 将对话历史设为对话历史的结果。

        query, rewritten = answerer.rewrite("why does it work?", conversation)  # 将查询、改写标记设为改写的结果。

        assert query == "why does it work?"  # 确认查询等于“why does it work?”。
        assert rewritten is False  # 确认改写标记为假。

    def test_empty_rewrite_falls_back_to_original(self):  # 验证改写响应为空时沿用原问题。
        generator = FakeGenerator(rewrite="   ")  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。
        conversation = Conversation(turns=[Turn("q", "a")])  # 将对话历史设为对话历史的结果。

        query, rewritten = answerer.rewrite("it?", conversation)  # 将查询、改写标记设为改写的结果。

        assert query == "it?"  # 确认查询等于“it?”。
        assert rewritten is False  # 确认改写标记为假。

    def test_overlong_rewrite_is_rejected(self):  # 验证过长的改写结果会被舍弃。
        generator = FakeGenerator(rewrite="x" * 700)  # 将生成器设为模拟生成器的结果。
        answerer = Answerer(_settings(), generator)  # 将问答服务设为问答服务的结果。
        conversation = Conversation(turns=[Turn("q", "a")])  # 将对话历史设为对话历史的结果。

        query, rewritten = answerer.rewrite("it?", conversation)  # 将查询、改写标记设为改写的结果。

        assert query == "it?"  # 确认查询等于“it?”。
        assert rewritten is False  # 确认改写标记为假。


class TestErrorDescriptions:  # 定义 TestErrorDescriptions 类型。
    def test_configuration_error_is_actionable(self):  # 验证配置错误带有可操作的提示。
        message = describe_generation_error(ConfigurationIssue("llm", "bad key"))  # 将消息设为整理生成错误提示的结果。

        assert "misconfigured" in message.lower()  # 确认“misconfigured”包含在消息的小写形式。

    def test_unreachable_error_mentions_service(self):  # 验证无法连接时提示故障服务。
        from rag_demo.errors import Unavailable  # 从 rag_demo.errors 导入服务不可达异常。

        message = describe_generation_error(Unavailable("llm", "connection refused"))  # 将消息设为整理生成错误提示的结果。

        assert "unreachable" in message.lower()  # 确认“unreachable”包含在消息的小写形式。

    def test_unknown_error_is_still_descriptive(self):  # 验证未知异常仍有可读的错误说明。
        message = describe_generation_error(RuntimeError("boom"))  # 将消息设为整理生成错误提示的结果。

        assert "boom" in message  # 确认“boom”包含在消息。