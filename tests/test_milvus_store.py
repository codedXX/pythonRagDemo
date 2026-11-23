"""Integration tests against the real Milvus instance.

These are skipped when no server is reachable, so the suite still runs offline.
They exist because the behaviour they check cannot be mocked convincingly: the
consistency-level trap is a property of the server, not of our code.
"""

from __future__ import annotations

import os
import uuid

import pytest

from rag_demo.config import load_settings
from rag_demo.store.milvus_store import ChildUnit, MilvusStore, ParentUnit, normalise

pytestmark = pytest.mark.integration


def _milvus_available(uri: str) -> bool:
    try:
        from pymilvus import MilvusClient

        MilvusClient(uri=uri).get_server_version()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    settings = load_settings(root=root, use_env=False)
    if not _milvus_available(settings.milvus.uri):
        pytest.skip(f"no Milvus server at {settings.milvus.uri}")
    # Unique names per test module so a stray run cannot touch other data.
    suffix = uuid.uuid4().hex[:6]
    settings.milvus.child_collection = f"test_child_{suffix}"
    settings.milvus.parent_collection = f"test_parent_{suffix}"
    settings.milvus.meta_collection = f"test_meta_{suffix}"
    return settings


@pytest.fixture(scope="module")
def store(settings):
    """Collections are created once per module.

    Creating them takes about nine seconds because index building is synchronous
    on the server, so doing it per test would dominate the suite's runtime.
    """
    store = MilvusStore(settings)
    store.drop_all()
    store.ensure_collections(dimension=8)
    yield store
    store.drop_all()


@pytest.fixture(autouse=True)
def clean_data(store):
    """Each test starts from empty collections so counts are meaningful."""
    for name in (store.child_name, store.parent_name):
        try:
            store.client.delete(name, filter="doc_id != \"\"")
        except Exception:
            pass
    try:
        store.client.delete(store.meta_name, filter='key != ""')
    except Exception:
        pass
    yield


def _vector(seed: float) -> list[float]:
    return normalise([seed, seed + 1, seed + 2, seed + 3, seed + 4, seed + 5, seed + 6, seed + 7])


def _child(doc_id: str, idx: int, seed: float = 1.0) -> ChildUnit:
    return ChildUnit(
        id=f"{doc_id}_c{idx}",
        doc_id=doc_id,
        parent_id=f"{doc_id}_p{idx // 2}",
        child_idx=idx,
        text=f"child text {idx}",
        breadcrumb="doc > chapter",
        page=idx,
        source_name="doc.pdf",
        embedding=_vector(seed),
    )


def _parent(doc_id: str, idx: int) -> ParentUnit:
    return ParentUnit(
        parent_id=f"{doc_id}_p{idx}",
        doc_id=doc_id,
        text=f"parent text {idx}",
        breadcrumb="doc > chapter",
        page_start=idx,
        source_name="doc.pdf",
    )


def test_inserted_rows_are_visible_immediately_without_flush(store):
    """The reason consistency_level is pinned to Strong.

    Under the default Bounded level this assertion fails: count is 0 and search
    returns nothing until an explicit flush.
    """
    store.insert_units([_child("d1", 0), _child("d1", 1)], [_parent("d1", 0)])
    store.activate_document("d1")

    stats = store.collection_stats()

    assert stats[store.child_name] == 2
    assert stats[store.parent_name] == 1
    hits = store.search_children(_vector(1.0), limit=5)
    assert len(hits) == 2


def test_inactive_rows_are_not_retrievable(store):
    store.insert_units([_child("d2", 0)], [_parent("d2", 0)])
    # Deliberately not activated, simulating an ingestion that failed partway.

    assert store.search_children(_vector(1.0), limit=5) == []
    assert store.all_active_children() == []


def test_activation_makes_document_retrievable(store):
    store.insert_units([_child("d3", 0)], [_parent("d3", 0)])
    assert store.search_children(_vector(1.0), limit=5) == []

    store.activate_document("d3")

    assert len(store.search_children(_vector(1.0), limit=5)) == 1


def test_deactivation_hides_document_again(store):
    store.insert_units([_child("d4", 0)], [_parent("d4", 0)])
    store.activate_document("d4")
    assert len(store.search_children(_vector(1.0), limit=5)) == 1

    store.deactivate_document("d4")

    assert store.search_children(_vector(1.0), limit=5) == []


def test_delete_removes_all_traces(store):
    store.insert_units([_child("d5", 0), _child("d5", 1)], [_parent("d5", 0), _parent("d5", 1)])
    store.activate_document("d5")

    store.delete_document("d5")

    assert store.search_children(_vector(1.0), limit=5) == []
    assert store.collection_stats()[store.child_name] == 0
    assert store.fetch_parents(["d5_p0"]) == {}


def test_fetch_parents_returns_requested_units(store):
    store.insert_units(
        [_child("d6", 0), _child("d6", 1)], [_parent("d6", 0), _parent("d6", 1)]
    )
    store.activate_document("d6")

    parents = store.fetch_parents(["d6_p0", "d6_p1", "missing"])

    assert set(parents) == {"d6_p0", "d6_p1"}
    assert parents["d6_p0"]["text"] == "parent text 0"


def test_search_respects_doc_filter(store):
    store.insert_units([_child("d7", 0)], [_parent("d7", 0)])
    store.insert_units([_child("d8", 0)], [_parent("d8", 0)])
    store.activate_document("d7")
    store.activate_document("d8")

    hits = store.search_children(_vector(1.0), limit=10, doc_id="d7")

    assert [h["doc_id"] for h in hits] == ["d7"]


def test_meta_round_trip(store):
    assert store.read_meta() == {}

    store.write_meta({"embedding_model": "m1", "dimension": 8})

    assert store.read_meta() == {"embedding_model": "m1", "dimension": 8}


def test_meta_is_replaced_not_appended(store):
    store.write_meta({"embedding_model": "m1", "dimension": 8})
    store.write_meta({"embedding_model": "m2", "dimension": 16})

    assert store.read_meta() == {"embedding_model": "m2", "dimension": 16}


def test_reinserting_same_document_does_not_duplicate(store):
    """Re-uploading a document must replace it, not accumulate copies."""
    store.insert_units([_child("d9", 0)], [_parent("d9", 0)])
    store.activate_document("d9")

    store.delete_document("d9")
    store.insert_units([_child("d9", 0)], [_parent("d9", 0)])
    store.activate_document("d9")

    assert len(store.search_children(_vector(1.0), limit=10)) == 1
    assert store.collection_stats()[store.child_name] == 1


def test_collections_use_strong_consistency(store):
    described = store.client.describe_collection(store.child_name)

    assert described.get("consistency_level") in {"Strong", 0, "0"}