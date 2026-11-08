from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
from qdrant_client import models

from mezvuat.persistence.blobstore import LocalBlobStore
from mezvuat.persistence.qdrant import build_filter, collection_name, point_id
from mezvuat.retrieval.types import Filters


def test_blob_store_is_content_addressed(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    digest = store.put(b"hello")
    assert digest == hashlib.sha256(b"hello").hexdigest()
    assert store.get(digest) == b"hello"
    assert store.exists(digest)
    assert store.put(b"hello") == digest
    store.delete(digest)
    assert not store.exists(digest)


def test_blob_store_rejects_path_tricks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalBlobStore(tmp_path).get("../../etc/passwd")


def test_filter_always_scopes_to_the_workspace() -> None:
    result = build_filter("ws-1", Filters(exclude_section_kinds=frozenset()))
    assert result.must is not None
    condition = result.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "workspace_id"
    assert condition.match == models.MatchValue(value="ws-1")


def test_filter_requires_a_workspace() -> None:
    with pytest.raises(ValueError):
        build_filter("", Filters())


def test_filter_translates_all_options() -> None:
    filters = Filters(
        document_ids=frozenset({"b", "a"}),
        latest_only=True,
        published_from=date(2024, 1, 1),
        published_to=date(2024, 12, 31),
    )
    result = build_filter("ws", filters)
    keys = [c.key for c in result.must or [] if isinstance(c, models.FieldCondition)]
    assert keys == ["workspace_id", "document_id", "is_latest_version", "published_at"]
    assert result.must_not is not None


def test_point_ids_are_deterministic_uuids() -> None:
    assert point_id("abc") == point_id("abc")
    assert point_id("abc") != point_id("abd")


def test_collection_names_are_safe() -> None:
    assert collection_name("BAAI/bge-small-en-v1.5", 384, "v1") == "chunks__bge-small-en-v1.5__384__v1"
