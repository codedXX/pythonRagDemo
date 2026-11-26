"""连接真实 Milvus 检验集合结构、强一致性和文档替换；服务不可达时跳过集成用例。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import os  # 引入操作系统接口。
import uuid  # 引入唯一标识生成器。

import pytest  # 引入测试框架。

from rag_demo.config import load_settings  # 从 rag_demo.config 导入加载配置。
from rag_demo.chunking.chunker import Child, Parent
from rag_demo.store.milvus_store import MilvusStore, normalise  # 从 rag_demo.store.milvus_store 引入本模块使用的类型和工具。

pytestmark = pytest.mark.integration  # 将pytestmark设为pytest.mark.integration。


def _milvus_available(uri: str) -> bool:  # 探测本机测试数据库是否可连接。
    try:  # 在此捕获后续操作可能产生的异常。
        from pymilvus import MilvusClient  # 从 pymilvus 导入数据库客户端。

        MilvusClient(uri=uri).get_server_version()  # 读取数据库版本。
        return True  # 返回真。
    except Exception:  # 捕获通用异常。
        return False  # 返回假。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def settings(tmp_path_factory):  # 取得当前使用的配置。
    root = tmp_path_factory.mktemp("store")  # 将项目根目录设为mktemp的结果。
    settings = load_settings(root=root, use_env=False)  # 将配置设为加载配置的结果。
    if not _milvus_available(settings.milvus.uri):  # 当不满足“检查测试数据库连接的结果”时进入此分支。
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")  # 跳过缺少外部依赖的测试。
    # 为本模块使用独立集合名，隔离其他测试数据。
    suffix = uuid.uuid4().hex[:6]  # 将扩展名设为uuid.uuid4().hex的指定区间。
    settings.milvus.child_collection = f"test_child_{suffix}"  # 将子块集合名设为格式化文本。
    settings.milvus.parent_collection = f"test_parent_{suffix}"  # 将父块集合名设为格式化文本。
    settings.milvus.meta_collection = f"test_meta_{suffix}"  # 将元数据集合名设为格式化文本。
    return settings  # 返回配置。


@pytest.fixture(scope="module")  # 注册供用例复用的测试夹具。
def store(settings):  # 每个模块只建一次集合，避免同步建索引的开销在每个用例中重复。
    """每个模块只建一次集合，避免同步建索引的开销在每个用例中重复。"""
    store = MilvusStore(settings)  # 将向量库设为向量库客户端的结果。
    store.drop_all()  # 清理测试集合。
    store.ensure_collections(dimension=8)  # 准备集合。
    yield store  # 向调用方逐项提供向量库。
    store.drop_all()  # 清理测试集合。


@pytest.fixture(autouse=True)  # 注册供用例复用的测试夹具。
def clean_data(store):  # 每个用例从空集合开始，保证计数有明确基准。
    """每个用例从空集合开始，保证计数有明确基准。"""
    for name in (store.child_name, store.parent_name):  # 遍历子块集合名、父块集合名，逐项取得名称。
        try:  # 在此捕获后续操作可能产生的异常。
            store.client.delete(name, filter="doc_id != \"\"")  # 向名称发送删除请求。
        except Exception:  # 捕获通用异常。
            pass  # 保留空实现。
    try:  # 在此捕获后续操作可能产生的异常。
        store.client.delete(store.meta_name, filter='key != ""')  # 向元数据集合名发送删除请求。
    except Exception:  # 捕获通用异常。
        pass  # 保留空实现。
    yield  # 向调用方逐项提供空值。


def _vector(seed: float) -> list[float]:  # 生成供测试使用的确定性向量。
    return normalise([seed, seed + 1, seed + 2, seed + 3, seed + 4, seed + 5, seed + 6, seed + 7])  # 返回归一化的结果。


def _child(doc_id: str, idx: int) -> Child:  # 定义子块的处理逻辑。
    return Child(  # 返回子块记录的结果。
        id=f"{doc_id}_c{idx}",  # 传入标识参数。
        doc_id=doc_id,  # 传入文档标识参数。
        parent_id=f"{doc_id}_p{idx // 2}",  # 传入所属父块标识参数。
        child_idx=idx,  # 传入子块序号参数。
        text=f"child text {idx}",  # 传入文本参数。
        breadcrumb="doc > chapter",  # 传入标题路径参数。
        page=idx,  # 传入页码参数。
        source_name="doc.pdf",  # 传入来源文件名参数。
    )  # 结束当前多行表达式。


def _parent(doc_id: str, idx: int) -> Parent:  # 定义父块的处理逻辑。
    return Parent(  # 返回父块记录的结果。
        parent_id=f"{doc_id}_p{idx}",  # 传入所属父块标识参数。
        doc_id=doc_id,  # 传入文档标识参数。
        text=f"parent text {idx}",  # 传入文本参数。
        breadcrumb="doc > chapter",  # 传入标题路径参数。
        index=idx,
        page_start=idx,  # 传入起始页参数。
        source_name="doc.pdf",  # 传入来源文件名参数。
    )  # 结束当前多行表达式。


def test_inserted_rows_are_visible_immediately_without_flush(store):  # 验证强一致性使写入立即可见。
    """验证强一致性使写入立即可见。默认的有界一致性可能要等到刷新后才能查到记录。"""
    store.insert_units([_child("d1", 0), _child("d1", 1)], [_parent("d1", 0)], [_vector(1.0)] * 2)  # 写入父子块。
    store.activate_document("d1")  # 激活文档。

    stats = store.collection_stats()  # 将统计信息设为统计集合记录的结果。

    assert stats[store.child_name] == 2  # 确认统计信息中对应的项等于2。
    assert stats[store.parent_name] == 1  # 确认统计信息中对应的项等于1。
    hits = store.search_children(_vector(1.0), limit=5)  # 将命中项设为检索子块的结果。
    assert len(hits) == 2  # 确认命中项的数量等于2。


def test_inactive_rows_are_not_retrievable(store):  # 验证未激活记录不能被召回。
    store.insert_units([_child("d2", 0)], [_parent("d2", 0)], [_vector(1.0)])  # 写入父子块。
    # 保留未激活状态，模拟入库中途失败。

    assert store.search_children(_vector(1.0), limit=5) == []  # 确认检索子块的结果等于空列表。
    assert store.all_active_children() == []  # 确认读取有效子块的结果等于空列表。


def test_activation_makes_document_retrievable(store):  # 验证激活后文档可以检索。
    store.insert_units([_child("d3", 0)], [_parent("d3", 0)], [_vector(1.0)])  # 写入父子块。
    assert store.search_children(_vector(1.0), limit=5) == []  # 确认检索子块的结果等于空列表。

    store.activate_document("d3")  # 激活文档。

    assert len(store.search_children(_vector(1.0), limit=5)) == 1  # 确认检索子块的结果的数量等于1。


def test_deactivation_hides_document_again(store):  # 验证取消激活后文档被隐藏。
    store.insert_units([_child("d4", 0)], [_parent("d4", 0)], [_vector(1.0)])  # 写入父子块。
    store.activate_document("d4")  # 激活文档。
    assert len(store.search_children(_vector(1.0), limit=5)) == 1  # 确认检索子块的结果的数量等于1。

    store.deactivate_document("d4")  # 隐藏文档片段。

    assert store.search_children(_vector(1.0), limit=5) == []  # 确认检索子块的结果等于空列表。


def test_delete_removes_all_traces(store):  # 验证删除后无残留片段。
    store.insert_units([_child("d5", 0), _child("d5", 1)], [_parent("d5", 0), _parent("d5", 1)], [_vector(1.0)] * 2)  # 写入父子块。
    store.activate_document("d5")  # 激活文档。

    store.delete_document("d5")  # 删除文档。

    assert store.search_children(_vector(1.0), limit=5) == []  # 确认检索子块的结果等于空列表。
    assert store.collection_stats()[store.child_name] == 0  # 确认统计集合记录的结果中对应的项等于0。
    assert store.fetch_parents(["d5_p0"]) == {}  # 确认读取父块的结果等于空字典。


def test_fetch_parents_returns_requested_units(store):  # 验证按主键只返回所请求的父块。
    store.insert_units(  # 写入父子块。
        [_child("d6", 0), _child("d6", 1)], [_parent("d6", 0), _parent("d6", 1)], [_vector(1.0)] * 2  # 为父块提供输入。
    )  # 结束当前多行表达式。
    store.activate_document("d6")  # 激活文档。

    parents = store.fetch_parents(["d6_p0", "d6_p1", "missing"])  # 将父块列表设为读取父块的结果。

    assert set(parents) == {"d6_p0", "d6_p1"}  # 确认去重集合等于集合。
    assert parents["d6_p0"]["text"] == "parent text 0"  # 确认父块列表中的d6_p0中的文本等于“parent text 0”。


def test_search_respects_doc_filter(store):  # 验证检索遵循文档过滤条件。
    store.insert_units([_child("d7", 0)], [_parent("d7", 0)], [_vector(1.0)])  # 写入父子块。
    store.insert_units([_child("d8", 0)], [_parent("d8", 0)], [_vector(1.0)])  # 写入父子块。
    store.activate_document("d7")  # 激活文档。
    store.activate_document("d8")  # 激活文档。

    hits = store.search_children(_vector(1.0), limit=10, doc_id="d7")  # 将命中项设为检索子块的结果。

    assert [h["doc_id"] for h in hits] == ["d7"]  # 确认从命中项中提取的各项等于列表。


def test_meta_round_trip(store):  # 验证元数据写入后可读回。
    assert store.read_meta() == {}  # 确认读取建库元数据的结果等于空字典。

    store.write_meta({"embedding_model": "m1", "dimension": 8})  # 保存建库元数据。

    assert store.read_meta() == {"embedding_model": "m1", "dimension": 8}  # 确认读取建库元数据的结果等于字段映射。


def test_meta_is_replaced_not_appended(store):  # 验证再次写入元数据会覆盖旧值。
    store.write_meta({"embedding_model": "m1", "dimension": 8})  # 保存建库元数据。
    store.write_meta({"embedding_model": "m2", "dimension": 16})  # 保存建库元数据。

    assert store.read_meta() == {"embedding_model": "m2", "dimension": 16}  # 确认读取建库元数据的结果等于字段映射。


def test_reinserting_same_document_does_not_duplicate(store):  # 重复上传应替换旧文件，避免累积副本。
    """重复上传应替换旧文件，避免累积副本。"""
    store.insert_units([_child("d9", 0)], [_parent("d9", 0)], [_vector(1.0)])  # 写入父子块。
    store.activate_document("d9")  # 激活文档。

    store.delete_document("d9")  # 删除文档。
    store.insert_units([_child("d9", 0)], [_parent("d9", 0)], [_vector(1.0)])  # 写入父子块。
    store.activate_document("d9")  # 激活文档。

    assert len(store.search_children(_vector(1.0), limit=10)) == 1  # 确认检索子块的结果的数量等于1。
    assert store.collection_stats()[store.child_name] == 1  # 确认统计集合记录的结果中对应的项等于1。


def test_collections_use_strong_consistency(store):  # 验证集合使用强一致性。
    described = store.client.describe_collection(store.child_name)  # 将described设为describe_collection的结果。

    assert described.get("consistency_level") in {"Strong", 0, "0"}  # 确认described中的一致性级别包含在集合。