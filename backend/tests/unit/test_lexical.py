from __future__ import annotations

import math
from pathlib import Path

import pytest

from mezvuat.retrieval.lexical import Bm25Index, WorkspaceLexicalIndexStore, payload_allowed, tokenize
from mezvuat.retrieval.types import ChunkPayload, Filters


def payload(
    chunk_id: str,
    *,
    document_id: str = "doc",
    latest: bool | None = True,
    published: str | None = None,
    kind: str = "body",
) -> ChunkPayload:
    return ChunkPayload(chunk_id, document_id, f"{document_id}-v", latest, published, kind)


def lucene_score(
    tf: float, doc_len: int, avg_len: float, n_docs: int, df: int, k1: float = 1.2, b: float = 0.75
) -> float:
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf * tf / (tf + k1 * (1 - b + b * doc_len / avg_len))


def test_scores_match_hand_computed_lucene_bm25() -> None:
    texts = ["retrieval evaluation benchmark", "retrieval retrieval pipeline", "unrelated cooking recipe"]
    index = Bm25Index.build([payload(str(i)) for i in range(3)], texts)
    hits = {h.chunk_id: h.score for h in index.search("retrieval", 3)}
    assert set(hits) == {"0", "1"}
    # Analysed lengths: 3, 3 and 3 stems; "retrieval" appears once in doc 0, twice in doc 1.
    assert hits["0"] == pytest.approx(lucene_score(1, 3, 3.0, 3, 2), rel=1e-4)
    assert hits["1"] == pytest.approx(lucene_score(2, 3, 3.0, 3, 2), rel=1e-4)


def test_stemming_matches_inflected_forms() -> None:
    index = Bm25Index.build([payload("a"), payload("b")], ["reranking passages", "gardening tips"])
    assert [h.chunk_id for h in index.search("rerank passage", 2)] == ["a"]


def test_query_of_only_stopwords_returns_nothing() -> None:
    index = Bm25Index.build([payload("a")], ["some content here"])
    assert tokenize(["the and of"]) == [[]]
    assert index.search("the and of", 5) == []


def test_filters_are_applied_before_top_k() -> None:
    payloads = [payload("refs", kind="references"), payload("body")]
    index = Bm25Index.build(payloads, ["bm25 bm25 bm25 ranking", "bm25 ranking overview text"])
    assert [h.chunk_id for h in index.search("bm25", 1)] == ["body"]
    everything = index.search("bm25", 1, Filters(exclude_section_kinds=frozenset()))
    assert [h.chunk_id for h in everything] == ["refs"]


def test_filter_semantics() -> None:
    filters = Filters(latest_only=True)
    assert payload_allowed(payload("a", latest=True), filters)
    assert not payload_allowed(payload("a", latest=False), filters)
    # Unknown "latest" status does not count as latest.
    assert not payload_allowed(payload("a", latest=None), filters)


def test_unknown_date_never_satisfies_a_date_filter() -> None:
    from datetime import date

    filters = Filters(published_from=date(2024, 1, 1))
    assert not payload_allowed(payload("a", published=None), filters)
    assert payload_allowed(payload("a", published="2024-06-01"), filters)
    assert not payload_allowed(payload("a", published="2023-06-01"), filters)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = Bm25Index.build([payload("a"), payload("b")], ["alpha beta gamma", "delta epsilon"])
    index.save(tmp_path / "idx")
    loaded = Bm25Index.load(tmp_path / "idx")
    assert [h.chunk_id for h in loaded.search("alpha", 2)] == ["a"]


def test_workspace_store_isolates_workspaces(tmp_path: Path) -> None:
    versions = {"ws-a": 1, "ws-b": 1}
    store = WorkspaceLexicalIndexStore(tmp_path, versions.get)
    store.write("ws-a", 1, [payload("a1")], ["confidential alpha report"])
    store.write("ws-b", 1, [payload("b1")], ["public beta notes"])
    assert [h.chunk_id for h in store.search("ws-a", "alpha", 5, Filters())] == ["a1"]
    assert store.search("ws-b", "alpha", 5, Filters()) == []
    assert store.search("ws-missing", "alpha", 5, Filters()) == []


def test_new_version_replaces_the_loaded_index(tmp_path: Path) -> None:
    versions = {"ws": 1}
    store = WorkspaceLexicalIndexStore(tmp_path, versions.get)
    store.write("ws", 1, [payload("old")], ["alpha content"])
    assert [h.chunk_id for h in store.search("ws", "alpha", 5, Filters())] == ["old"]
    store.write("ws", 2, [payload("new")], ["alpha content"])
    versions["ws"] = 2
    assert [h.chunk_id for h in store.search("ws", "alpha", 5, Filters())] == ["new"]
    store.prune("ws", 2)
    assert not (tmp_path / "ws" / "v1").exists()
