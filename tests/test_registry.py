"""Tests for the SQLite document registry."""

from __future__ import annotations

import pytest

from rag_demo.store.registry import DocumentRegistry, DocumentState


@pytest.fixture()
def registry(tmp_path):
    return DocumentRegistry(tmp_path / "rag.db")


def test_begin_creates_staging_record(registry):
    record = registry.begin("d1", "paper.pdf")

    assert record.state is DocumentState.STAGING
    assert record.usable is False
    assert record.child_count == 0


def test_mark_active_records_counts(registry):
    registry.begin("d1", "paper.pdf")

    registry.mark_active("d1", parent_count=12, child_count=48, char_count=9000)

    record = registry.get("d1")
    assert record.state is DocumentState.ACTIVE
    assert record.usable is True
    assert (record.parent_count, record.child_count, record.char_count) == (12, 48, 9000)


def test_mark_failed_records_reason(registry):
    registry.begin("d1", "paper.pdf")

    registry.mark_failed("d1", "PDF appears to be scanned")

    record = registry.get("d1")
    assert record.state is DocumentState.FAILED
    assert record.usable is False
    assert "scanned" in record.error


def test_failed_document_is_excluded_from_usable(registry):
    registry.begin("ok", "good.pdf")
    registry.mark_active("ok", 1, 2, 300)
    registry.begin("bad", "bad.pdf")
    registry.mark_failed("bad", "boom")

    usable = registry.list_usable()

    assert [d.doc_id for d in usable] == ["ok"]


def test_list_documents_includes_failed(registry):
    """A failed upload must stay visible so its reason can be shown."""
    registry.begin("bad", "bad.pdf")
    registry.mark_failed("bad", "scanned")

    listed = registry.list_documents()

    assert [d.doc_id for d in listed] == ["bad"]
    assert listed[0].state is DocumentState.FAILED


def test_reingesting_same_id_resets_state(registry):
    registry.begin("d1", "paper.pdf")
    registry.mark_active("d1", 1, 2, 300)

    record = registry.begin("d1", "paper.pdf")

    assert record.state is DocumentState.STAGING
    assert record.child_count == 0
    assert record.error is None


def test_remove_deletes_record(registry):
    registry.begin("d1", "paper.pdf")

    registry.remove("d1")

    assert registry.get("d1") is None


def test_find_by_name_finds_duplicates(registry):
    registry.begin("a", "same.pdf")
    registry.begin("b", "same.pdf")

    found = registry.find_by_name("same.pdf")

    assert {d.doc_id for d in found} == {"a", "b"}


def test_counts_group_by_state(registry):
    registry.begin("a", "a.pdf")
    registry.mark_active("a", 1, 1, 10)
    registry.begin("b", "b.pdf")
    registry.mark_failed("b", "x")

    counts = registry.counts()

    assert counts == {DocumentState.ACTIVE.value: 1, DocumentState.FAILED.value: 1}


def test_survives_reopen(tmp_path):
    path = tmp_path / "rag.db"
    first = DocumentRegistry(path)
    first.begin("d1", "paper.pdf")
    first.mark_active("d1", 3, 9, 100)

    reopened = DocumentRegistry(path)

    record = reopened.get("d1")
    assert record is not None
    assert record.child_count == 9


def test_as_dict_is_serialisable(registry):
    import json

    registry.begin("d1", "paper.pdf")
    registry.mark_active("d1", 1, 2, 3)

    payload = json.loads(json.dumps(registry.get("d1").as_dict()))

    assert payload["state"] == "active"