"""Reciprocal Rank Fusion (Cormack, Clarke, Buettcher, SIGIR 2009); see ADR-0006."""

from __future__ import annotations

from collections.abc import Sequence

from mezvuat.retrieval.types import ScoredChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[ScoredChunk]], *, k: int = DEFAULT_RRF_K) -> list[ScoredChunk]:
    """Fuse ranked lists using 1-based ranks. Equal scores break deterministically on chunk id."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [ScoredChunk(chunk_id=chunk_id, score=score) for chunk_id, score in fused]
