"""Retrieval value types and the protocols the pipeline depends on."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol

RetrievalMode = Literal["dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank"]


@dataclass(frozen=True)
class Filters:
    """Metadata filters. An empty value means no constraint on that field."""

    document_ids: frozenset[str] = frozenset()
    latest_only: bool = False
    published_from: date | None = None
    published_to: date | None = None
    exclude_section_kinds: frozenset[str] = frozenset({"references"})


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class ChunkPayload:
    """Fields both indices need in order to apply `Filters`. Dates are ISO strings or None when unknown."""

    chunk_id: str
    document_id: str
    document_version_id: str
    is_latest_version: bool | None
    published_at: str | None
    section_kind: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    workspace_id: str
    document_id: str
    document_version_id: str
    text: str
    text_sha256: str
    embed_text: str
    ordinal: int
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    section_kind: str
    is_latest_version: bool | None = None
    document_title: str | None = None
    version_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedEvidence:
    chunk: ChunkRecord
    rank: int
    score: float
    lexical_rank: int | None
    dense_rank: int | None
    fusion_score: float | None
    rerank_score: float | None


@dataclass
class RetrievalResult:
    mode: RetrievalMode
    evidence: list[RankedEvidence]
    timings_ms: dict[str, float]


class LexicalRetriever(Protocol):
    def search(self, workspace_id: str, query: str, k: int, filters: Filters) -> list[ScoredChunk]: ...


class DenseRetriever(Protocol):
    def search(self, workspace_id: str, query_vector: list[float], k: int, filters: Filters) -> list[ScoredChunk]: ...


class ChunkStore(Protocol):
    def get_many(self, workspace_id: str, chunk_ids: list[str]) -> dict[str, ChunkRecord]: ...


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class PassageReranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...
