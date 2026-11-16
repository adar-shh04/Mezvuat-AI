"""API request and response models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from mezvuat.orchestration import FinalAnswer


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


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["dense", "bm25", "hybrid_rrf", "hybrid_rrf_rerank"] = "hybrid_rrf"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    # Answer as of this date: per document, only the version shown to apply on that date is used (ADR-0010).
    as_of: date | None = None


class RunCreated(BaseModel):
    run_id: str
    status: str
    run_url: str
    events_url: str


class RunErrorOut(BaseModel):
    code: str
    message: str | None


class LlmCallOut(BaseModel):
    client_request_id: str
    workflow_node: str | None
    task_type: str
    backend: str
    provider: str | None
    model: str | None
    status: str | None
    error_code: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    # ModelMesh correlation (contract §1) and the reason a direct backend answered instead. None on direct calls.
    modelmesh_request_id: str | None = None
    modelmesh_trace_id: str | None = None
    fallback_reason: str | None = None
    cache_status: str | None = None
    # ModelMesh's estimate, not billing data.
    cost_estimated_usd: float | None = None


class RunSummaryOut(BaseModel):
    """Run counters (`api.runs.summarize`). Empty when a run crashed before producing a final state."""

    model_config = ConfigDict(extra="allow")

    as_of: str | None = None
    iterations: int | None = None
    resynthesized: bool | None = None
    evidence_items: int | None = None
    retrieval_calls: int | None = None
    llm_calls: int | None = None
    llm_calls_left: int | None = None
    # The backend that served the run's calls: "direct", "modelmesh", or "modelmesh+direct" when some fell back.
    backend: str | None = None
    backends: list[str] = Field(default_factory=list)
    # True when at least one call was answered by a direct backend because ModelMesh could not serve it.
    fallback: bool | None = None
    fallback_reasons: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RunOut(BaseModel):
    run_id: str
    workspace_id: str
    question: str
    status: Literal["queued", "running", "completed", "failed"]
    data_policy: Literal["any", "local_only"]
    retrieval_mode: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: RunErrorOut | None
    # Markdown, citations (exact spans and pages), claim verdicts, limitations and flags.
    answer: FinalAnswer | None
    summary: RunSummaryOut
    llm_calls: list[LlmCallOut]


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["up", "down"]
    # Optional target inside the answer: a claim id from `answer.claims`. Unset means the whole answer.
    claim_id: str | None = Field(default=None, max_length=40)
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackOut(BaseModel):
    feedback_id: str
    run_id: str
    rating: Literal["up", "down"]
    claim_id: str | None
    created_at: datetime


class VersionTextOut(BaseModel):
    """One page of a version's normalized text. Citation offsets are absolute, so the span on this page is
    `text[char_start - start : char_end - start]`."""

    document_id: str
    document_version_id: str
    page: int | None
    page_count: int
    start: int
    end: int
    text: str
