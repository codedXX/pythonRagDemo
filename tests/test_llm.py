"""验证生成适配器的流式输出、错误分类和启动探测。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import json  # 引入结构化数据的编解码工具。

import httpx  # 引入网络客户端库。
import pytest  # 引入测试框架。

from rag_demo.config import LLMSettings  # 从 rag_demo.config 导入生成模型配置。
from rag_demo.errors import ConfigurationIssue, ContractMismatch, TransientFailure  # 从 rag_demo.errors 引入本模块使用的类型和工具。
from rag_demo.generation.llm import Generator, Message  # 从 rag_demo.generation.llm 引入本模块使用的类型和工具。

URL = "https://relay.example.com/v1/chat/completions"  # 设置URL。


def _settings(**overrides) -> LLMSettings:  # 在测试默认配置上应用本用例的覆盖项。
    base = dict(  # 将基础值设为按字段构造的字典。
        base_url="https://relay.example.com/v1",  # 传入服务根地址参数。
        api_key="sk-test",  # 传入接口密钥参数。
        model="claude-sonnet-5",  # 传入模型参数。
        timeout_seconds=5.0,  # 传入超时秒数参数。
    )  # 结束当前多行表达式。
    base.update(overrides)  # 用覆盖项更新基础值。
    return LLMSettings(**base)  # 返回生成模型配置的结果。


class _FakeStreamResponse:  # 模拟网络客户端流式响应的上下文管理器。
    """模拟网络客户端流式响应的上下文管理器。"""

    def __init__(self, status_code: int, lines: list[str], body: str = "") -> None:  # 保存实例所需的配置和依赖。
        self.status_code = status_code  # 保存传入的响应状态码。
        self._lines = lines  # 保存传入的文本行列表。
        self._body = body  # 保存传入的正文。

    def __enter__(self):  # 定义__enter__的处理逻辑。
        return self  # 返回当前实例。

    def __exit__(self, *exc):  # 定义__exit__的处理逻辑。
        return False  # 返回假。

    def iter_lines(self):  # 定义iter_lines的处理逻辑。
        yield from self._lines  # 向调用方逐项提供文本行列表。

    def read(self):  # 定义读取内容的处理逻辑。
        return self._body.encode()  # 返回文本编码的结果。


def _sse(pieces: list[str]) -> list[str]:  # 定义推送事件的处理逻辑。
    lines = []  # 初始化文本行列表为空列表。
    for piece in pieces:  # 遍历片段列表，逐项取得片段。
        lines.append(  # 将“data: ”加上dumps的结果加入文本行列表。
            "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})  # 填入内容字段。
        )  # 结束当前多行表达式。
        lines.append("")  # 将空字符串加入文本行列表。
    lines.append("data: [DONE]")  # 将“data: [DONE]”加入文本行列表。
    return lines  # 返回文本行列表。


@pytest.fixture(autouse=True)  # 注册供用例复用的测试夹具。
def _no_network(monkeypatch):  # 拦截本组测试中意外发起的外部请求。
    def explode(*args, **kwargs):  # 定义explode的处理逻辑。
        raise AssertionError("unexpected network call in test")  # 向上抛出断言异常。

    monkeypatch.setattr(httpx, "stream", explode)  # 临时替换网络客户端库的“stream”以控制测试依赖。
    monkeypatch.setattr(httpx, "get", explode)  # 临时替换网络客户端库的“get”以控制测试依赖。


def test_streams_pieces_incrementally(monkeypatch):  # 验证生成片段能逐段送达。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["Hel", "lo", " world"]))  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    pieces = list(Generator(_settings()).stream_chat([Message("user", "hi")]))  # 将片段列表设为流式生成回复的结果的列表形式。

    assert pieces == ["Hel", "lo", " world"]  # 确认片段列表等于列表。
    assert "".join(pieces) == "Hello world"  # 确认拼接后的片段列表等于“Hello world”。


def test_stream_request_asks_for_streaming(monkeypatch):  # 验证请求显式开启流式传输。
    captured: dict = {}  # 初始化捕获的数据为空字典。

    def capture(method, url, json=None, headers=None, timeout=None):  # 定义capture的处理逻辑。
        captured["json"] = json  # 将捕获的数据中的结构化数据设为结构化数据。
        captured["url"] = url  # 将捕获的数据中的服务地址设为服务地址。
        return _FakeStreamResponse(200, _sse(["ok"]))  # 返回模拟流式响应的结果。

    monkeypatch.setattr(httpx, "stream", capture)  # 临时替换网络客户端库的“stream”以控制测试依赖。
    list(Generator(_settings()).stream_chat([Message("user", "hi")]))  # 列表。

    assert captured["url"] == URL  # 确认捕获的数据中的服务地址等于URL。
    assert captured["json"]["stream"] is True  # 确认捕获的数据中的结构化数据中的数据流为真。
    assert captured["json"]["stream_options"] == {"include_usage": False}  # 确认捕获的数据中的结构化数据中的stream_options等于字段映射。
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]  # 确认捕获的数据中的结构化数据中的消息列表等于列表。


def test_complete_joins_chunks(monkeypatch):  # 验证完整回答由片段拼接而成。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["a", "b"]))  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    assert Generator(_settings()).complete([Message("user", "x")]) == "ab"  # 确认获取完整回答的结果等于“ab”。


def test_tolerates_message_shaped_chunks(monkeypatch):  # 验证兼容从消息字段返回文本的响应。
    lines = [  # 将文本行列表设为列表。
        "data: " + json.dumps({"choices": [{"message": {"content": "alt"}}]}),  # 填入内容字段。
        "data: [DONE]",  # 模拟供应商发来的流式结束标记。
    ]  # 结束列表。
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, lines))  # 临时替换网络客户端库的“stream”以控制测试依赖。

    assert Generator(_settings()).complete([Message("user", "x")]) == "alt"  # 确认获取完整回答的结果等于“alt”。


def test_skips_non_data_lines(monkeypatch):  # 验证流解析跳过非数据行。
    lines = [": keep-alive", "", "data: " + json.dumps({"choices": []}), "data: [DONE]"]  # 将文本行列表设为列表。
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, lines))  # 临时替换网络客户端库的“stream”以控制测试依赖。

    assert Generator(_settings()).complete([Message("user", "x")]) == ""  # 确认获取完整回答的结果等于空字符串。


def test_missing_api_key_is_configuration_issue():  # 验证缺少密钥抛出配置异常。
    with pytest.raises(ConfigurationIssue):  # 验证下方操作会抛出供应商配置异常。
        list(Generator(_settings(api_key="")).stream_chat([Message("user", "x")]))  # 列表。


def test_401_is_configuration_issue(monkeypatch):  # 验证认证失败归为配置错误。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "stream",  # 替换建立流式连接的方法以检查异常路径。
        lambda *a, **k: _FakeStreamResponse(  # 用匿名回调模拟调用结果。
            401, [], json.dumps({"error": {"message": "Incorrect API key provided."}})  # 为dumps提供输入。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        list(Generator(_settings()).stream_chat([Message("user", "x")]))  # 列表。

    assert "credentials" in str(excinfo.value)  # 确认“credentials”包含在值的字符串形式。


def test_unknown_model_names_the_model(monkeypatch):  # 验证未知模型的提示包含模型名。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "stream",  # 替换建立流式连接的方法以检查异常路径。
        lambda *a, **k: _FakeStreamResponse(  # 用匿名回调模拟调用结果。
            400, [], json.dumps({"error": {"message": "model not found"}})  # 为dumps提供输入。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        list(Generator(_settings(model="claude-sonnet-5")).stream_chat([Message("user", "x")]))  # 列表。

    assert "claude-sonnet-5" in str(excinfo.value)  # 确认“claude-sonnet-5”包含在值的字符串形式。


def test_retry_when_nothing_emitted_yet(monkeypatch):  # 验证尚未输出时允许重试。
    attempts = {"n": 0}  # 将总尝试次数设为字段映射。

    def flaky(method, url, json=None, headers=None, timeout=None):  # 定义flaky的处理逻辑。
        attempts["n"] += 1  # 将1累加到总尝试次数中的n。
        if attempts["n"] == 1:  # 当总尝试次数中的n等于1时进入此分支。
            return _FakeStreamResponse(503, [], "")  # 返回模拟流式响应的结果。
        return _FakeStreamResponse(200, _sse(["ok"]))  # 返回模拟流式响应的结果。

    monkeypatch.setattr(httpx, "stream", flaky)  # 临时替换网络客户端库的“stream”以控制测试依赖。

    text = Generator(_settings(), sleep=lambda s: None).complete([Message("user", "x")])  # 将文本设为获取完整回答的结果。

    assert text == "ok"  # 确认文本等于“ok”。
    assert attempts["n"] == 2  # 确认总尝试次数中的n等于2。


def test_does_not_retry_after_emitting(monkeypatch):  # 若在输出途中重试，会让用户收到重复文本。
    """若在输出途中重试，会让用户收到重复文本。"""
    attempts = {"n": 0}  # 将总尝试次数设为字段映射。

    class FailsAfterOneChunk(_FakeStreamResponse):  # 定义 FailsAfterOneChunk 类型。
        def iter_lines(self):  # 定义iter_lines的处理逻辑。
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "partial"}}]})  # 向调用方逐项提供“data: ”加上dumps的结果。
            raise TransientFailure("llm", "connection dropped")  # 向上抛出暂时性故障。

    def flaky(method, url, json=None, headers=None, timeout=None):  # 定义flaky的处理逻辑。
        attempts["n"] += 1  # 将1累加到总尝试次数中的n。
        return FailsAfterOneChunk(200, [])  # 返回FailsAfterOneChunk的结果。

    monkeypatch.setattr(httpx, "stream", flaky)  # 临时替换网络客户端库的“stream”以控制测试依赖。
    seen: list[str] = []  # 初始化已处理项为空列表。

    with pytest.raises(TransientFailure):  # 验证下方操作会抛出暂时性故障。
        for piece in Generator(_settings(), sleep=lambda s: None).stream_chat([Message("user", "x")]):  # 遍历流式生成回复的结果，逐项取得片段。
            seen.append(piece)  # 将片段加入已处理项。

    assert seen == ["partial"]  # 确认已处理项等于列表。
    assert attempts["n"] == 1  # 确认总尝试次数中的n等于1。


def test_bad_json_chunk_is_contract_mismatch(monkeypatch):  # 验证无效的流式数据归为协议错误。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, ["data: {not json"])  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        list(Generator(_settings()).stream_chat([Message("user", "x")]))  # 列表。


def test_verify_rejects_empty_answer(monkeypatch):  # 验证自检拒绝空回答。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["   "]))  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        Generator(_settings()).verify()  # 自检开关。


def test_verify_returns_text(monkeypatch):  # 验证自检返回实际生成的文本。
    monkeypatch.setattr(  # 临时替换网络客户端库的“stream”以控制测试依赖。
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(200, _sse(["ok"]))  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    assert Generator(_settings()).verify() == "ok"  # 确认自检开关的结果等于“ok”。


def test_list_models_returns_ids(monkeypatch):  # 验证模型列表接口返回标识列表。
    response = httpx.Response(  # 将响应设为网络响应的结果。
        200,  # 传入200。
        json={"data": [{"id": "claude-sonnet-5"}, {"id": "gpt-4o"}]},  # 传入结构化数据参数。
        request=httpx.Request("GET", "https://relay.example.com/v1/models"),  # 传入请求参数。
    )  # 结束当前多行表达式。
    monkeypatch.setattr(httpx, "get", lambda *a, **k: response)  # 临时替换网络客户端库的“get”以控制测试依赖。

    assert Generator(_settings()).list_models() == ["claude-sonnet-5", "gpt-4o"]  # 确认列出模型的结果等于列表。