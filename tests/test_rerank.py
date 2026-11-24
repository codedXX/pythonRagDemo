"""验证百炼原生重排协议所用的请求体、结果层级及错误结构。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import httpx  # 引入网络客户端库。
import pytest  # 引入测试框架。

from rag_demo.config import RerankSettings  # 从 rag_demo.config 导入重排配置。
from rag_demo.embedding.rerank import RerankResult, Reranker  # 从 rag_demo.embedding.rerank 引入本模块使用的类型和工具。
from rag_demo.errors import ConfigurationIssue, ContractMismatch, TransientFailure  # 从 rag_demo.errors 引入本模块使用的类型和工具。

URL = "https://dashscope.example.com/api/v1/services/rerank/text-rerank/text-rerank"  # 设置URL。


def _settings(**overrides) -> RerankSettings:  # 在测试默认配置上应用本用例的覆盖项。
    base = dict(  # 将基础值设为按字段构造的字典。
        base_url="https://dashscope.example.com",  # 传入服务根地址参数。
        path="/api/v1/services/rerank/text-rerank/text-rerank",  # 传入路径参数。
        api_key="sk-test",  # 传入接口密钥参数。
        model="test-rerank",  # 传入模型参数。
        max_retries=3,  # 传入最大尝试次数参数。
    )  # 结束当前多行表达式。
    base.update(overrides)  # 用覆盖项更新基础值。
    return RerankSettings(**base)  # 返回重排配置的结果。


def _ok(results: list[dict]) -> httpx.Response:  # 定义成功标志的处理逻辑。
    return httpx.Response(  # 返回网络响应的结果。
        200,  # 传入200。
        json={"output": {"results": results}, "usage": {"total_tokens": 12}},  # 传入结构化数据参数。
        request=httpx.Request("POST", URL),  # 传入请求参数。
    )  # 结束当前多行表达式。


@pytest.fixture(autouse=True)  # 注册供用例复用的测试夹具。
def _no_network(monkeypatch):  # 拦截本组测试中意外发起的外部请求。
    def explode(*args, **kwargs):  # 定义explode的处理逻辑。
        raise AssertionError("unexpected network call in test")  # 向上抛出断言异常。

    monkeypatch.setattr(httpx, "post", explode)  # 临时替换网络客户端库的“post”以控制测试依赖。


def test_url_is_the_native_path_not_the_compatible_one():  # 验证重排请求使用百炼原生路径。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。

    # 确认使用百炼原生重排路径。
    assert reranker.settings.url == URL  # 确认服务地址等于URL。
    assert "/compatible-mode/" not in reranker.settings.url  # 确认“/compatible-mode/”未包含在服务地址。


def test_parses_native_envelope_and_sorts_by_score(monkeypatch):  # 验证解析原生结果并按分数排序。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: _ok(  # 用匿名回调模拟调用结果。
            [  # 开始列出各项。
                {"index": 0, "relevance_score": 0.10},  # 为成功标志提供输入。
                {"index": 2, "relevance_score": 0.95},  # 为成功标志提供输入。
                {"index": 1, "relevance_score": 0.42},  # 为成功标志提供输入。
            ]  # 结束列表。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    results = reranker.rerank("q", ["a", "b", "c"])  # 将结果列表设为重排的结果。

    assert [r.index for r in results] == [2, 1, 0]  # 确认从结果列表中提取的各项等于列表。
    assert results[0].score == pytest.approx(0.95)  # 确认分数等于approx的结果。


def test_sends_the_native_request_shape(monkeypatch):  # 验证提交的数据符合原生协议。
    captured: dict = {}  # 初始化捕获的数据为空字典。

    def capture(url, json=None, headers=None, timeout=None):  # 定义capture的处理逻辑。
        captured["url"] = url  # 将捕获的数据中的服务地址设为服务地址。
        captured["json"] = json  # 将捕获的数据中的结构化数据设为结构化数据。
        return _ok([{"index": 0, "relevance_score": 0.5}])  # 返回成功标志的结果。

    monkeypatch.setattr(httpx, "post", capture)  # 临时替换网络客户端库的“post”以控制测试依赖。
    Reranker(_settings()).rerank("query text", ["doc one", "doc two"])  # 重排。

    assert captured["url"] == URL  # 确认捕获的数据中的服务地址等于URL。
    # 查询和候选列表应封装在原生协议的 input 对象中。
    assert captured["json"]["input"] == {  # 确认捕获的数据中的结构化数据中的输入等于字段映射。
        "query": "query text",  # 填入查询字段。
        "documents": ["doc one", "doc two"],  # 填入文档字段。
    }  # 结束本组字段。
    assert captured["json"]["model"] == "test-rerank"  # 确认捕获的数据中的结构化数据中的模型等于“test-rerank”。


def test_rejects_openai_shaped_response(monkeypatch):  # 验证拒绝其他协议的结果结构。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(  # 用匿名回调模拟调用结果。
            200,  # 传入200。
            json={"data": [{"index": 0, "relevance_score": 0.5}]},  # 传入结构化数据参数。
            request=httpx.Request("POST", URL),  # 传入请求参数。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch) as excinfo:  # 验证下方操作会抛出响应协议异常。
        reranker.rerank("q", ["a"])  # 重排。

    assert "output" in str(excinfo.value)  # 确认“output”包含在值的字符串形式。


def test_rejects_out_of_range_index(monkeypatch):  # 验证拒绝越界的原始下标。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx, "post", lambda *a, **k: _ok([{"index": 7, "relevance_score": 0.9}])  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        reranker.rerank("q", ["a"])  # 重排。


def test_rejects_duplicate_indices(monkeypatch):  # 验证拒绝重复下标。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: _ok(  # 用匿名回调模拟调用结果。
            [  # 开始列出各项。
                {"index": 0, "relevance_score": 0.9},  # 为成功标志提供输入。
                {"index": 0, "relevance_score": 0.4},  # 为成功标志提供输入。
            ]  # 结束列表。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        reranker.rerank("q", ["a", "b"])  # 重排。


def test_rejects_non_numeric_score(monkeypatch):  # 验证拒绝非数值相关性分数。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx, "post", lambda *a, **k: _ok([{"index": 0, "relevance_score": "high"}])  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        reranker.rerank("q", ["a"])  # 重排。


def test_404_explains_the_compatible_path_trap(monkeypatch):  # 验证路径不存在时提示正确的原生接口。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(404, json={}, request=httpx.Request("POST", URL)),  # 用匿名回调模拟调用结果。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        reranker.rerank("q", ["a"])  # 重排。

    message = str(excinfo.value)  # 将消息设为值的字符串形式。
    assert "native path" in message  # 确认“native path”包含在消息。
    assert "compatible" in message  # 确认“compatible”包含在消息。


def test_invalid_key_uses_dashscope_error_shape(monkeypatch):  # 验证使用原生错误结构解释凭据问题。
    reranker = Reranker(_settings())  # 将重排器设为重排服务的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(  # 用匿名回调模拟调用结果。
            401,  # 传入401。
            json={"code": "InvalidApiKey", "message": "Invalid API-key provided."},  # 传入结构化数据参数。
            request=httpx.Request("POST", URL),  # 传入请求参数。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        reranker.rerank("q", ["a"])  # 重排。

    # 保留供应商返回的错误码，使调用方能定位具体原因。
    assert "InvalidApiKey" in str(excinfo.value)  # 确认“InvalidApiKey”包含在值的字符串形式。


def test_missing_api_key_is_configuration_issue():  # 验证缺少密钥抛出配置异常。
    with pytest.raises(ConfigurationIssue):  # 验证下方操作会抛出供应商配置异常。
        Reranker(_settings(api_key="")).rerank("q", ["a"])  # 重排。


def test_retries_then_succeeds(monkeypatch):  # 验证临时故障经重试可以恢复。
    attempts = {"n": 0}  # 将总尝试次数设为字段映射。

    def flaky(url, json=None, headers=None, timeout=None):  # 定义flaky的处理逻辑。
        attempts["n"] += 1  # 将1累加到总尝试次数中的n。
        if attempts["n"] == 1:  # 当总尝试次数中的n等于1时进入此分支。
            return httpx.Response(503, json={}, request=httpx.Request("POST", URL))  # 返回网络响应的结果。
        return _ok([{"index": 0, "relevance_score": 0.8}])  # 返回成功标志的结果。

    monkeypatch.setattr(httpx, "post", flaky)  # 临时替换网络客户端库的“post”以控制测试依赖。
    results = Reranker(_settings(), sleep=lambda s: None).rerank("q", ["a"])  # 将结果列表设为重排的结果。

    assert results[0].index == 0  # 确认索引等于0。
    assert attempts["n"] == 2  # 确认总尝试次数中的n等于2。


def test_transient_exhaustion_raises(monkeypatch):  # 验证重试次数耗尽后抛出异常。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(503, json={}, request=httpx.Request("POST", URL)),  # 用匿名回调模拟调用结果。
    )  # 结束当前多行表达式。

    with pytest.raises(TransientFailure):  # 验证下方操作会抛出暂时性故障。
        Reranker(_settings(max_retries=2), sleep=lambda s: None).rerank("q", ["a"])  # 重排。


def test_empty_documents_makes_no_request(monkeypatch):  # 验证候选列表为空时跳过网络请求。
    assert Reranker(_settings()).rerank("q", []) == []  # 确认重排的结果等于空列表。


def test_verify_accepts_sensible_ranking(monkeypatch):  # 验证重排自检接受符合预期的顺序。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: _ok(  # 用匿名回调模拟调用结果。
            [  # 开始列出各项。
                {"index": 0, "relevance_score": 0.91},  # 为成功标志提供输入。
                {"index": 1, "relevance_score": 0.05},  # 为成功标志提供输入。
                {"index": 2, "relevance_score": 0.02},  # 为成功标志提供输入。
            ]  # 结束列表。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    results = Reranker(_settings()).verify()  # 将结果列表设为自检开关的结果。

    assert results[0].index == 0  # 确认索引等于0。


def test_verify_rejects_implausible_ranking(monkeypatch):  # 验证重排自检拒绝明显错误的顺序。
    # 让服务返回格式正确但顺序相反的结果，验证自检能发现排序异常。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: _ok(  # 用匿名回调模拟调用结果。
            [  # 开始列出各项。
                {"index": 2, "relevance_score": 0.99},  # 为成功标志提供输入。
                {"index": 0, "relevance_score": 0.01},  # 为成功标志提供输入。
            ]  # 结束列表。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch) as excinfo:  # 验证下方操作会抛出响应协议异常。
        Reranker(_settings()).verify()  # 自检开关。

    assert "implausible" in str(excinfo.value)  # 确认“implausible”包含在值的字符串形式。


def test_top_n_is_sent_when_requested(monkeypatch):  # 验证指定条数时将上限传给服务端。
    captured: dict = {}  # 初始化捕获的数据为空字典。

    def capture(url, json=None, headers=None, timeout=None):  # 定义capture的处理逻辑。
        captured["json"] = json  # 将捕获的数据中的结构化数据设为结构化数据。
        return _ok([{"index": 0, "relevance_score": 0.5}])  # 返回成功标志的结果。

    monkeypatch.setattr(httpx, "post", capture)  # 临时替换网络客户端库的“post”以控制测试依赖。
    Reranker(_settings()).rerank("q", ["a", "b"], top_n=1)  # 重排。

    assert captured["json"]["parameters"] == {"top_n": 1}  # 确认捕获的数据中的结构化数据中的parameters等于字段映射。