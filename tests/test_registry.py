"""验证 SQLite 文档登记及状态变更。"""

from __future__ import annotations  # 延后解析类型注解，允许引用尚未定义的类型。

import pytest  # 引入测试框架。

from rag_demo.store.registry import DocumentRegistry, DocumentState  # 从 rag_demo.store.registry 引入本模块使用的类型和工具。


@pytest.fixture()  # 注册供用例复用的测试夹具。
def registry(tmp_path):  # 定义登记表的处理逻辑。
    return DocumentRegistry(tmp_path / "rag.db")  # 返回文档登记表的结果。


def test_begin_creates_staging_record(registry):  # 验证登记后文档处于入库中状态。
    record = registry.begin("d1", "paper.pdf")  # 将登记记录设为登记待入库文档的结果。

    assert record.state is DocumentState.STAGING  # 确认状态为DocumentState.STAGING。
    assert record.usable is False  # 确认可用状态为假。
    assert record.child_count == 0  # 确认子块数等于0。


def test_mark_active_records_counts(registry):  # 验证激活时记录父子块数量。
    registry.begin("d1", "paper.pdf")  # 登记待入库文档。

    registry.mark_active("d1", parent_count=12, child_count=48, char_count=9000)  # 标记文档可用。

    record = registry.get("d1")  # 将登记记录设为读取的结果。
    assert record.state is DocumentState.ACTIVE  # 确认状态为DocumentState.ACTIVE。
    assert record.usable is True  # 确认可用状态为真。
    assert (record.parent_count, record.child_count, record.char_count) == (12, 48, 9000)  # 确认父块数、子块数、字符数等于组合值。


def test_mark_failed_records_reason(registry):  # 验证失败时保存原因。
    registry.begin("d1", "paper.pdf")  # 登记待入库文档。

    registry.mark_failed("d1", "PDF appears to be scanned")  # 登记失败原因。

    record = registry.get("d1")  # 将登记记录设为读取的结果。
    assert record.state is DocumentState.FAILED  # 确认状态为DocumentState.FAILED。
    assert record.usable is False  # 确认可用状态为假。
    assert "scanned" in record.error  # 确认“scanned”包含在错误。


def test_failed_document_is_excluded_from_usable(registry):  # 验证失败文档排除在可用列表之外。
    registry.begin("ok", "good.pdf")  # 登记待入库文档。
    registry.mark_active("ok", 1, 2, 300)  # 标记文档可用。
    registry.begin("bad", "bad.pdf")  # 登记待入库文档。
    registry.mark_failed("bad", "boom")  # 登记失败原因。

    usable = registry.list_usable()  # 将可用状态设为列出有效文档的结果。

    assert [d.doc_id for d in usable] == ["ok"]  # 确认从可用状态中提取的各项等于列表。


def test_list_documents_includes_failed(registry):  # 保留上传失败的记录，供界面显示失败原因。
    """保留上传失败的记录，供界面显示失败原因。"""
    registry.begin("bad", "bad.pdf")  # 登记待入库文档。
    registry.mark_failed("bad", "scanned")  # 登记失败原因。

    listed = registry.list_documents()  # 将listed设为列出登记文档的结果。

    assert [d.doc_id for d in listed] == ["bad"]  # 确认从listed中提取的各项等于列表。
    assert listed[0].state is DocumentState.FAILED  # 确认状态为DocumentState.FAILED。


def test_reingesting_same_id_resets_state(registry):  # 验证同一标识重新入库会重置状态。
    registry.begin("d1", "paper.pdf")  # 登记待入库文档。
    registry.mark_active("d1", 1, 2, 300)  # 标记文档可用。

    record = registry.begin("d1", "paper.pdf")  # 将登记记录设为登记待入库文档的结果。

    assert record.state is DocumentState.STAGING  # 确认状态为DocumentState.STAGING。
    assert record.child_count == 0  # 确认子块数等于0。
    assert record.error is None  # 确认错误为空值。


def test_remove_deletes_record(registry):  # 验证删除操作移除登记行。
    registry.begin("d1", "paper.pdf")  # 登记待入库文档。

    registry.remove("d1")  # 移除。

    assert registry.get("d1") is None  # 确认读取的结果为空值。


def test_find_by_name_finds_duplicates(registry):  # 验证按名称可查询同名记录。
    registry.begin("a", "same.pdf")  # 登记待入库文档。
    registry.begin("b", "same.pdf")  # 登记待入库文档。

    found = registry.find_by_name("same.pdf")  # 将命中通道列表设为按名称查询的结果。

    assert {d.doc_id for d in found} == {"a", "b"}  # 确认从命中通道列表中提取的各项等于集合。


def test_counts_group_by_state(registry):  # 验证计数按状态分组。
    registry.begin("a", "a.pdf")  # 登记待入库文档。
    registry.mark_active("a", 1, 1, 10)  # 标记文档可用。
    registry.begin("b", "b.pdf")  # 登记待入库文档。
    registry.mark_failed("b", "x")  # 登记失败原因。

    counts = registry.counts()  # 将按状态计数设为按状态计数的结果。

    assert counts == {DocumentState.ACTIVE.value: 1, DocumentState.FAILED.value: 1}  # 确认按状态计数等于字段映射。


def test_survives_reopen(tmp_path):  # 验证重新打开数据库后记录仍在。
    path = tmp_path / "rag.db"  # 将路径设为临时目录下的“rag.db”。
    first = DocumentRegistry(path)  # 将首项设为文档登记表的结果。
    first.begin("d1", "paper.pdf")  # 登记待入库文档。
    first.mark_active("d1", 3, 9, 100)  # 标记文档可用。

    reopened = DocumentRegistry(path)  # 将reopened设为文档登记表的结果。

    record = reopened.get("d1")  # 将登记记录设为reopened中的d1。
    assert record is not None  # 确认登记记录不为空值。
    assert record.child_count == 9  # 确认子块数等于9。


def test_as_dict_is_serialisable(registry):  # 验证记录转换的字典可以序列化。
    import json  # 引入结构化数据的编解码工具。

    registry.begin("d1", "paper.pdf")  # 登记待入库文档。
    registry.mark_active("d1", 1, 2, 3)  # 标记文档可用。

    payload = json.loads(json.dumps(registry.get("d1").as_dict()))  # 将载荷设为loads的结果。

    assert payload["state"] == "active"  # 确认载荷中的状态等于“active”。