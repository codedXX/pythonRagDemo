"""验证向量响应解析、错误分类、分批请求和磁盘缓存。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import httpx  # 引入网络客户端库。
import pytest  # 引入测试框架。

from rag_demo.config import EmbeddingSettings  # 从 rag_demo.config 导入向量服务配置。
from rag_demo.embedding.provider import (  # 从 rag_demo.embedding.provider 引入本模块使用的类型和工具。
    EmbeddingCache,  # 导入磁盘向量缓存。
    EmbeddingClient,  # 导入向量请求客户端。
    Embedder,  # 导入向量编码器。
    content_hash,  # 导入文本与模型的缓存摘要。
)  # 结束当前多行表达式。
from rag_demo.errors import ConfigurationIssue, ContractMismatch, TransientFailure  # 从 rag_demo.errors 引入本模块使用的类型和工具。

DIM = 8  # 将测试向量维数设为8。


def _settings(**overrides) -> EmbeddingSettings:  # 在测试默认配置上应用本用例的覆盖项。
    base = dict(  # 将基础值设为按字段构造的字典。
        base_url="https://example.invalid/v1",  # 传入服务根地址参数。
        api_key="sk-test",  # 传入接口密钥参数。
        model="test-embedding",  # 传入模型参数。
        batch_size=2,  # 传入单批文本数参数。
        max_retries=3,  # 传入最大尝试次数参数。
        retry_base_delay_seconds=0.0,  # 传入初始退避秒数参数。
    )  # 结束当前多行表达式。
    base.update(overrides)  # 用覆盖项更新基础值。
    return EmbeddingSettings(**base)  # 返回向量服务配置的结果。


def _reply(vectors: list[list[float]], indices: list[int] | None = None) -> httpx.Response:  # 组装供客户端解析的模拟响应。
    order = indices if indices is not None else list(range(len(vectors)))  # 将次序设为按条件选取的值。
    payload = {  # 将载荷设为字段映射。
        "object": "list",  # 填入对象类型字段。
        "data": [  # 填入数据字段。
            {"object": "embedding", "index": i, "embedding": v}  # 组合当前结果所需的各项数据。
            for i, v in zip(order, vectors, strict=True)  # 逐项处理推导式中的i, v。
        ],  # 结束列表。
    }  # 结束本组字段。
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://x"))  # 返回网络响应的结果。


@pytest.fixture(autouse=True)  # 注册供用例复用的测试夹具。
def _no_network(monkeypatch):  # 意外发起真实网络请求时立即使测试失败。
    """意外发起真实网络请求时立即使测试失败。"""

    def explode(*args, **kwargs):  # 定义explode的处理逻辑。
        raise AssertionError("unexpected network call in test")  # 向上抛出断言异常。

    monkeypatch.setattr(httpx, "post", explode)  # 临时替换网络客户端库的“post”以控制测试依赖。


def test_parse_returns_vectors_in_input_order(monkeypatch):  # 验证按输入顺序恢复向量。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    # 构造乱序响应，确认按原始下标把向量还原给对应子块。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx, "post", lambda *a, **k: _reply([[1.0] * DIM, [2.0] * DIM], indices=[1, 0])  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    vectors = client.embed(["first", "second"])  # 将向量列表设为生成向量的结果。

    assert vectors[0] == [2.0] * DIM  # 确认向量列表的首项等于列表乘以向量维度。
    assert vectors[1] == [1.0] * DIM  # 确认向量列表的第二项等于列表乘以向量维度。


def test_parse_rejects_length_mismatch(monkeypatch):  # 验证向量条数不符时拒绝响应。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply([[1.0] * DIM]))  # 临时替换网络客户端库的“post”以控制测试依赖。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        client.embed(["a", "b"])  # 生成向量。


def test_parse_rejects_missing_data(monkeypatch):  # 验证缺少数据字段时拒绝响应。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(200, json={"nope": True}, request=httpx.Request("POST", "https://x")),  # 用匿名回调模拟调用结果。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        client.embed(["a"])  # 生成向量。


def test_parse_rejects_inconsistent_dimensions(monkeypatch):  # 验证各向量维度不一致时拒绝响应。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx, "post", lambda *a, **k: _reply([[1.0] * DIM, [1.0] * (DIM + 1)])  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    with pytest.raises(ContractMismatch):  # 验证下方操作会抛出响应协议异常。
        client.embed(["a", "b"])  # 生成向量。


def test_missing_api_key_is_a_configuration_issue():  # 验证密钥缺失归为配置错误。
    client = EmbeddingClient(_settings(api_key=""))  # 将客户端设为向量请求客户端的结果。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        client.embed(["a"])  # 生成向量。

    assert "api_key" in str(excinfo.value)  # 确认“api_key”包含在值的字符串形式。


def test_401_is_configuration_not_transient(monkeypatch):  # 验证认证失败归为配置错误。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(  # 用匿名回调模拟调用结果。
            401,  # 传入401。
            json={"error": {"message": "Incorrect API key provided."}},  # 传入结构化数据参数。
            request=httpx.Request("POST", "https://x"),  # 传入请求参数。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        client.embed(["a"])  # 生成向量。

    assert "credentials" in str(excinfo.value)  # 确认“credentials”包含在值的字符串形式。


def test_unknown_model_400_is_configuration(monkeypatch):  # 验证模型不存在的请求错误归为配置问题。
    client = EmbeddingClient(_settings(model="klingon-embedding-9"))  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx,  # 传入网络客户端库。
        "post",  # 替换网络提交方法，使用测试指定的模拟响应。
        lambda *a, **k: httpx.Response(  # 用匿名回调模拟调用结果。
            400,  # 传入400。
            json={"error": {"message": "The model does not exist."}},  # 传入结构化数据参数。
            request=httpx.Request("POST", "https://x"),  # 传入请求参数。
        ),  # 结束当前多行表达式。
    )  # 结束当前多行表达式。

    with pytest.raises(ConfigurationIssue) as excinfo:  # 验证下方操作会抛出供应商配置异常。
        client.embed(["a"])  # 生成向量。

    # 错误提示应包含模型名，方便直接修改对应配置。
    assert "klingon-embedding-9" in str(excinfo.value)  # 确认“klingon-embedding-9”包含在值的字符串形式。


def test_429_is_transient(monkeypatch):  # 验证限流归为暂时性故障。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(  # 临时替换网络客户端库的“post”以控制测试依赖。
        httpx, "post", lambda *a, **k: httpx.Response(429, json={}, request=httpx.Request("POST", "https://x"))  # 替换网络请求入口，使用测试指定的响应。
    )  # 结束当前多行表达式。

    with pytest.raises(TransientFailure):  # 验证下方操作会抛出暂时性故障。
        client.embed(["a"])  # 生成向量。


def test_unreachable_is_reported_as_unavailable(monkeypatch):  # 验证连接失败标记为服务不可达。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。

    def boom(*args, **kwargs):  # 定义boom的处理逻辑。
        raise httpx.ConnectError("nope")  # 向上抛出httpx.ConnectError。

    monkeypatch.setattr(httpx, "post", boom)  # 临时替换网络客户端库的“post”以控制测试依赖。

    with pytest.raises(Exception) as excinfo:  # 验证下方操作会抛出通用异常。
        client.embed(["a"])  # 生成向量。

    assert "cannot reach" in str(excinfo.value)  # 确认“cannot reach”包含在值的字符串形式。


def test_embedder_batches_requests(monkeypatch):  # 验证按配置批量提交文本。
    client = EmbeddingClient(_settings(batch_size=2))  # 将客户端设为向量请求客户端的结果。
    calls: list[int] = []  # 初始化调用记录为空列表。

    def fake_post(url, json=None, headers=None, timeout=None):  # 定义fake_post的处理逻辑。
        calls.append(len(json["input"]))  # 将结构化数据中的输入的长度加入调用记录。
        return _reply([[0.5] * DIM for _ in json["input"]])  # 返回模拟向量响应的结果。

    monkeypatch.setattr(httpx, "post", fake_post)  # 临时替换网络客户端库的“post”以控制测试依赖。
    embedder = Embedder(_settings(batch_size=2), client=client)  # 将向量服务设为向量编码器的结果。

    vectors = embedder.embed_texts(["a", "b", "c", "d", "e"])  # 将向量列表设为批量生成文本向量的结果。

    assert len(vectors) == 5  # 确认向量列表的数量等于5。
    assert calls == [2, 2, 1]  # 确认调用记录等于列表。


def test_embedder_retries_transient_then_succeeds(monkeypatch):  # 验证临时故障重试后可成功生成向量。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    attempts = {"n": 0}  # 将总尝试次数设为字段映射。

    def flaky(url, json=None, headers=None, timeout=None):  # 定义flaky的处理逻辑。
        attempts["n"] += 1  # 将1累加到总尝试次数中的n。
        if attempts["n"] == 1:  # 当总尝试次数中的n等于1时进入此分支。
            return httpx.Response(503, json={}, request=httpx.Request("POST", "https://x"))  # 返回网络响应的结果。
        return _reply([[0.1] * DIM for _ in json["input"]])  # 返回模拟向量响应的结果。

    monkeypatch.setattr(httpx, "post", flaky)  # 临时替换网络客户端库的“post”以控制测试依赖。
    embedder = Embedder(_settings(max_retries=3), client=client, sleep=lambda s: None)  # 将向量服务设为向量编码器的结果。

    vectors = embedder.embed_texts(["a"])  # 将向量列表设为批量生成文本向量的结果。

    assert len(vectors) == 1  # 确认向量列表的数量等于1。
    assert attempts["n"] == 2  # 确认总尝试次数中的n等于2。


def test_embedder_does_not_retry_configuration_errors(monkeypatch):  # 验证错误配置无需重试。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    attempts = {"n": 0}  # 将总尝试次数设为字段映射。

    def deny(url, json=None, headers=None, timeout=None):  # 定义deny的处理逻辑。
        attempts["n"] += 1  # 将1累加到总尝试次数中的n。
        return httpx.Response(  # 返回网络响应的结果。
            401,  # 传入401。
            json={"error": {"message": "Incorrect API key provided."}},  # 传入结构化数据参数。
            request=httpx.Request("POST", "https://x"),  # 传入请求参数。
        )  # 结束当前多行表达式。

    monkeypatch.setattr(httpx, "post", deny)  # 临时替换网络客户端库的“post”以控制测试依赖。
    embedder = Embedder(_settings(max_retries=5), client=client, sleep=lambda s: None)  # 将向量服务设为向量编码器的结果。

    with pytest.raises(ConfigurationIssue):  # 验证下方操作会抛出供应商配置异常。
        embedder.embed_texts(["a"])  # 批量生成文本向量。

    # 凭据错误时应立即结束，避免重试消耗时间与配额。
    assert attempts["n"] == 1  # 确认总尝试次数中的n等于1。


def test_cache_avoids_second_request(monkeypatch, tmp_path):  # 验证缓存命中时省去重复请求。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    calls = {"n": 0}  # 将调用记录设为字段映射。

    def counting(url, json=None, headers=None, timeout=None):  # 定义counting的处理逻辑。
        calls["n"] += 1  # 将1累加到调用记录中的n。
        return _reply([[0.7] * DIM for _ in json["input"]])  # 返回模拟向量响应的结果。

    monkeypatch.setattr(httpx, "post", counting)  # 临时替换网络客户端库的“post”以控制测试依赖。
    cache = EmbeddingCache(tmp_path)  # 将缓存设为磁盘向量缓存的结果。
    embedder = Embedder(_settings(), client=client, cache=cache)  # 将向量服务设为向量编码器的结果。

    first = embedder.embed_texts(["hello"])  # 将首项设为批量生成文本向量的结果。
    second = embedder.embed_texts(["hello"])  # 将第二项设为批量生成文本向量的结果。

    assert first == second  # 确认首项等于第二项。
    assert calls["n"] == 1  # 确认调用记录中的n等于1。


def test_cache_miss_when_model_changes(monkeypatch, tmp_path):  # 验证切换模型造成缓存未命中。
    calls = {"n": 0}  # 将调用记录设为字段映射。

    def counting(url, json=None, headers=None, timeout=None):  # 定义counting的处理逻辑。
        calls["n"] += 1  # 将1累加到调用记录中的n。
        return _reply([[0.7] * DIM for _ in json["input"]])  # 返回模拟向量响应的结果。

    monkeypatch.setattr(httpx, "post", counting)  # 临时替换网络客户端库的“post”以控制测试依赖。
    cache = EmbeddingCache(tmp_path)  # 将缓存设为磁盘向量缓存的结果。

    Embedder(_settings(model="model-a"), client=EmbeddingClient(_settings(model="model-a")), cache=cache).embed_texts(["hi"])  # 批量生成文本向量。
    Embedder(_settings(model="model-b"), client=EmbeddingClient(_settings(model="model-b")), cache=cache).embed_texts(["hi"])  # 批量生成文本向量。

    # 不同模型的向量空间不同，因此缓存也必须区分模型。
    assert calls["n"] == 2  # 确认调用记录中的n等于2。


def test_content_hash_differs_per_model_and_text():  # 验证模型或文本改变都会影响缓存键。
    assert content_hash("a", "m") != content_hash("a", "n") != content_hash("b", "m")  # 确认文本与模型的缓存摘要的结果不等于文本与模型的缓存摘要的结果不等于文本与模型的缓存摘要的结果。
    assert content_hash("a", "m") == content_hash("a", "m")  # 确认文本与模型的缓存摘要的结果等于文本与模型的缓存摘要的结果。


def test_probe_dimension_reports_vector_length(monkeypatch):  # 验证探测维度等于返回向量长度。
    client = EmbeddingClient(_settings())  # 将客户端设为向量请求客户端的结果。
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _reply([[0.0] * 1536]))  # 临时替换网络客户端库的“post”以控制测试依赖。

    assert Embedder(_settings(), client=client).probe_dimension() == 1536  # 确认探测向量维度的结果等于1536。