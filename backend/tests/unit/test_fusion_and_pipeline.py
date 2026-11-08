from __future__ import annotations

import pytest

from mezvuat.retrieval.fusion import reciprocal_rank_fusion
from mezvuat.retrieval.pipeline import RetrievalConfig, RetrievalPipeline, collapse_cross_version_duplicates
from mezvuat.retrieval.types import ChunkRecord, Filters, ScoredChunk


def scored(*ids: str) -> list[ScoredChunk]:
    return [ScoredChunk(i, 1.0 / (n + 1)) for n, i in enumerate(ids)]


def record(chunk_id: str, *, doc: str = "d", sha: str | None = None, latest: bool | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        workspace_id="ws",
        document_id=doc,
        document_version_id=f"{doc}-v",
        text=f"text of {chunk_id}",
        text_sha256=sha or chunk_id,
        embed_text="",
        ordinal=0,
        char_start=0,
        char_end=1,
        page_start=None,
        page_end=None,
        section_path=(),
        section_kind="body",
        is_latest_version=latest,
    )


def test_rrf_matches_hand_computed_scores() -> None:
    fused = reciprocal_rank_fusion([scored("a", "b"), scored("b", "c")], k=60)
    assert [f.chunk_id for f in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)


def test_rrf_breaks_ties_on_chunk_id() -> None:
    fused = reciprocal_rank_fusion([scored("z"), scored("a")], k=60)
    assert [f.chunk_id for f in fused] == ["a", "z"]


def test_rrf_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([scored("a")], k=0)


def test_collapse_prefers_latest_version_at_best_rank() -> None:
    records = {
        "old": record("old", sha="same", latest=False),
        "new": record("new", sha="same", latest=True),
        "other": record("other", sha="different", latest=True),
    }
    assert collapse_cross_version_duplicates(["old", "other", "new"], records) == ["new", "other"]


def test_collapse_keeps_identical_text_from_different_documents() -> None:
    records = {"a": record("a", doc="d1", sha="x"), "b": record("b", doc="d2", sha="x")}
    assert collapse_cross_version_duplicates(["a", "b"], records) == ["a", "b"]


class FakeLexical:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[str, int, Filters]] = []

    def search(self, workspace_id: str, query: str, k: int, filters: Filters) -> list[ScoredChunk]:
        self.calls.append((workspace_id, k, filters))
        return scored(*self.ids)


class FakeDense:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.workspaces: list[str] = []

    def search(self, workspace_id: str, query_vector: list[float], k: int, filters: Filters) -> list[ScoredChunk]:
        self.workspaces.append(workspace_id)
        return scored(*self.ids)


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.0]


class FakeStore:
    def __init__(self, records: dict[str, ChunkRecord]) -> None:
        self.records = records

    def get_many(self, workspace_id: str, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        return {c: self.records[c] for c in chunk_ids if c in self.records}


class ReversingReranker:
    """Scores later passages higher, so a rerank visibly changes the order."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(i) for i in range(len(passages))]


def make_pipeline(lexical: list[str], dense: list[str], *, reranker: object | None = None) -> RetrievalPipeline:
    records = {c: record(c) for c in {*lexical, *dense}}
    return RetrievalPipeline(
        FakeLexical(lexical),
        FakeDense(dense),
        FakeEmbedder(),
        FakeStore(records),
        reranker=reranker,  # type: ignore[arg-type]
    )


def test_each_mode_uses_only_its_retrievers() -> None:
    bm25 = make_pipeline(["a", "b"], ["c"]).retrieve("ws", "q", k=5, config=RetrievalConfig(mode="bm25"))
    assert [e.chunk.chunk_id for e in bm25.evidence] == ["a", "b"]
    assert "dense" not in bm25.timings_ms

    dense = make_pipeline(["a"], ["c", "d"]).retrieve("ws", "q", k=5, config=RetrievalConfig(mode="dense"))
    assert [e.chunk.chunk_id for e in dense.evidence] == ["c", "d"]
    assert "lexical" not in dense.timings_ms


def test_hybrid_records_both_ranks() -> None:
    result = make_pipeline(["a", "b"], ["b", "c"]).retrieve("ws", "q", k=3, config=RetrievalConfig(mode="hybrid_rrf"))
    top = result.evidence[0]
    assert top.chunk.chunk_id == "b"
    assert (top.lexical_rank, top.dense_rank) == (2, 1)
    assert top.fusion_score is not None


def test_rerank_reorders_only_the_rerank_depth() -> None:
    pipeline = make_pipeline(["a", "b", "c", "d"], ["a", "b", "c", "d"], reranker=ReversingReranker())
    output = pipeline.retrieve("ws", "q", k=4, config=RetrievalConfig(mode="hybrid_rrf_rerank", rerank_depth=3))
    assert [e.chunk.chunk_id for e in output.evidence] == ["c", "b", "a", "d"]
    assert output.evidence[3].rerank_score is None


def test_rerank_mode_without_reranker_fails_loudly() -> None:
    with pytest.raises(ValueError):
        make_pipeline(["a"], ["a"]).retrieve("ws", "q", config=RetrievalConfig(mode="hybrid_rrf_rerank"))


def test_workspace_is_passed_to_both_retrievers() -> None:
    lexical = FakeLexical(["a"])
    dense = FakeDense(["a"])
    pipeline = RetrievalPipeline(lexical, dense, FakeEmbedder(), FakeStore({"a": record("a")}))
    pipeline.retrieve("ws-42", "q", config=RetrievalConfig(mode="hybrid_rrf"))
    assert lexical.calls[0][0] == "ws-42"
    assert dense.workspaces == ["ws-42"]
