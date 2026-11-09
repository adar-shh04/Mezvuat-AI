"""API request and response models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)


class WorkspaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    data_classification: str
    created_at: datetime


class UrlIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class TemporalFactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kind: str
    value: date
    precision: str
    source: str
    confidence: str
    evidence: str | None


class DocumentVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    version_label: str | None
    version_ordinal: int | None
    is_latest_version: bool | None
    raw_sha256: str | None
    text_sha256: str | None
    probable_duplicate_of: str | None
    source_type: str
    source_uri: str | None
    license: str | None
    public_corpus_verified: bool
    ingested_at: datetime
    ingestion_status: str
    error_code: str | None
    error_message: str | None
    extractor: str | None
    temporal_facts: list[TemporalFactOut]


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workspace_id: str
    canonical_key: str | None
    title: str | None
    authors: list[str] | None
    versions: list[DocumentVersionOut]


class SubmitOut(BaseModel):
    status: Literal["accepted", "duplicate"]
    version: DocumentVersionOut


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] = Field(default_factory=list)
    latest_only: bool = False
    published_from: date | None = None
    published_to: date | None = None
    include_references: bool = False


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=10, ge=1, le=100)
    mode: Literal["dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank"] = "hybrid_rrf"
    candidate_depth: int = Field(default=100, ge=1, le=1000)
    rerank_depth: int = Field(default=50, ge=1, le=200)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class EvidenceOut(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_version_id: str
    document_title: str | None
    version_label: str | None
    is_latest_version: bool | None
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    section_kind: str
    lexical_rank: int | None
    dense_rank: int | None
    fusion_score: float | None
    rerank_score: float | None


class SearchResponse(BaseModel):
    mode: str
    evidence: list[EvidenceOut]
    timings_ms: dict[str, float]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    postgres: str
    qdrant: str
