"""The retrieval pipeline shared by the API and the experiment harness (ADR-0006).

Filters are pushed down into both retrievers rather than applied after fusion, so a restrictive filter cannot
empty the candidate list; the result is the same set the spec's fuse-then-filter order describes, with better recall.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from mezvuat.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from mezvuat.retrieval.types import (
    ChunkRecord,
    ChunkStore,
    DenseRetriever,
    Filters,
    LexicalRetriever,
    PassageReranker,
    QueryEmbedder,
    RankedEvidence,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "hybrid_rrf_rerank"
    candidate_depth: int = 100
    rerank_depth: int = 50
    rrf_k: int = DEFAULT_RRF_K
    collapse_duplicates: bool = True

    def __post_init__(self) -> None:
        if self.candidate_depth <= 0 or self.rerank_depth <= 0:
            raise ValueError("depths must be positive")


def _rank_map(items: list[ScoredChunk]) -> dict[str, int]:
    return {item.chunk_id: rank for rank, item in enumerate(items, start=1)}


def collapse_cross_version_duplicates(ordered_ids: list[str], records: dict[str, ChunkRecord]) -> list[str]:
    """Keep one chunk per identical text within a document, preferring the latest version.

    The kept chunk takes the position of the best-ranked duplicate. Identical text across *different* documents is
    left alone: that is two sources saying the same thing, and both are citable.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for chunk_id in ordered_ids:
        record = records.get(chunk_id)
        if record is not None:
            groups.setdefault((record.document_id, record.text_sha256), []).append(chunk_id)
    winner_for: dict[tuple[str, str], str] = {}
    for key, members in groups.items():
        latest = [m for m in members if records[m].is_latest_version]
        winner_for[key] = latest[0] if latest else members[0]
    seen: set[tuple[str, str]] = set()
    collapsed: list[str] = []
    for chunk_id in ordered_ids:
        record = records.get(chunk_id)
        if record is None:
            continue
        key = (record.document_id, record.text_sha256)
        if key not in seen:
            seen.add(key)
            collapsed.append(winner_for[key])
    return collapsed


class RetrievalPipeline:
    def __init__(
        self,
        lexical: LexicalRetriever,
        dense: DenseRetriever,
        embedder: QueryEmbedder,
        store: ChunkStore,
        reranker: PassageReranker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._lexical = lexical
        self._dense = dense
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._clock = clock

    def _timed[T](self, timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
        start = self._clock()
        result = fn()
        timings[name] = (self._clock() - start) * 1000.0
        return result

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        *,
        k: int = 10,
        config: RetrievalConfig | None = None,
        filters: Filters | None = None,
    ) -> RetrievalResult:
        config = config or RetrievalConfig()
        filters = filters or Filters()
        timings: dict[str, float] = {}
        total_start = self._clock()
        uses_lexical = config.mode in ("bm25", "hybrid_rrf", "hybrid_rrf_rerank")
        uses_dense = config.mode in ("dense", "hybrid_rrf", "hybrid_rrf_rerank")
        hybrid = uses_lexical and uses_dense
        depth = max(config.candidate_depth, k)

        lexical: list[ScoredChunk] = []
        if uses_lexical:
            lexical = self._timed(timings, "lexical", lambda: self._lexical.search(workspace_id, query, depth, filters))
        dense: list[ScoredChunk] = []
        if uses_dense:
            vector = self._timed(timings, "embed_query", lambda: self._embedder.embed_query(query))
            dense = self._timed(timings, "dense", lambda: self._dense.search(workspace_id, vector, depth, filters))

        fusion_scores: dict[str, float] = {}
        if hybrid:
            fused = self._timed(timings, "fusion", lambda: reciprocal_rank_fusion([lexical, dense], k=config.rrf_k))
            fusion_scores = {item.chunk_id: item.score for item in fused}
        else:
            fused = lexical if uses_lexical else dense
        ordered = [item.chunk_id for item in fused]
        base_scores = {item.chunk_id: item.score for item in fused}

        records = self._timed(timings, "fetch", lambda: self._store.get_many(workspace_id, ordered))
        ordered = [chunk_id for chunk_id in ordered if chunk_id in records]
        if config.collapse_duplicates:
            ordered = collapse_cross_version_duplicates(ordered, records)

        rerank_scores: dict[str, float] = {}
        if config.mode == "hybrid_rrf_rerank":
            if self._reranker is None:
                raise ValueError("mode hybrid_rrf_rerank requires a reranker")
            reranker = self._reranker
            head = ordered[: config.rerank_depth]
            passages = [records[chunk_id].text for chunk_id in head]
            scores = self._timed(timings, "rerank", lambda: reranker.score(query, passages))
            rerank_scores = dict(zip(head, scores, strict=True))
            head_sorted = sorted(head, key=lambda cid: (-rerank_scores[cid], cid))
            ordered = head_sorted + ordered[config.rerank_depth :]

        lexical_ranks = _rank_map(lexical)
        dense_ranks = _rank_map(dense)
        evidence = [
            RankedEvidence(
                chunk=records[chunk_id],
                rank=rank,
                score=rerank_scores.get(chunk_id, base_scores[chunk_id]),
                lexical_rank=lexical_ranks.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
                fusion_score=fusion_scores.get(chunk_id),
                rerank_score=rerank_scores.get(chunk_id),
            )
            for rank, chunk_id in enumerate(ordered[:k], start=1)
        ]
        timings["total"] = (self._clock() - total_start) * 1000.0
        return RetrievalResult(mode=config.mode, evidence=evidence, timings_ms=timings)
