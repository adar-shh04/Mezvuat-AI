"""Research state (ADR-0008). List fields with reducers accumulate; everything else is last-write-wins."""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mezvuat.citations import EvidenceItem, PageSpan, SpanRef
from mezvuat.inference import DataPolicy
from mezvuat.orchestration.models import (
    Citation,
    ClaimVerdict,
    Coverage,
    Draft,
    DraftClaim,
    DraftQuote,
    FinalAnswer,
    QueryAnalysis,
    QueryRewrite,
    Reformulation,
    ResearchPlan,
    SubQuestion,
    Sufficiency,
    TemporalScope,
)
from mezvuat.temporal import AsOfResolution, DateFact, VersionDates

# Bounds (ADR-0008 "Bounds"). Run-level budgets come from settings; these are structural.
MAX_ITER = 2
MAX_RESYNTH = 1
MAX_EVIDENCE_FOR_SYNTHESIS = 20
RECURSION_LIMIT = 30
# LLM calls held back for synthesize + respond, so a research loop can never starve the answer.
RESERVED_CALLS = 2


def merge_evidence(left: dict[str, EvidenceItem], right: dict[str, EvidenceItem]) -> dict[str, EvidenceItem]:
    """Registry ids are assigned once and never rebound."""
    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key].chunk_id != value.chunk_id:
            raise ValueError(f"evidence id {key} is already bound to another chunk")
        merged.setdefault(key, value)
    return merged


def merge_documents(
    left: dict[str, list[VersionDates]], right: dict[str, list[VersionDates]]
) -> dict[str, list[VersionDates]]:
    return {**left, **right}


class RunFilters(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ids: list[str] = Field(default_factory=list)
    latest_only: bool = False
    published_from: date | None = None
    published_to: date | None = None
    include_references: bool = False


class PendingQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    subquestion_id: str
    query: str


class RetrievalCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: int
    subquestion_id: str
    query: str
    results: int
    new_evidence: int
    latency_ms: float


class LLMCallRecord(BaseModel):
    """One HTTP attempt to an LLM backend; mirrors an `llm_call` row."""

    model_config = ConfigDict(frozen=True)

    client_request_id: str
    run_id: str
    workflow_node: str
    task_type: str
    backend: str
    provider: str
    model: str | None
    prompt_version: str
    iteration: int
    status: str
    error_code: str | None
    http_status: int | None
    provider_request_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: str
    latency_ms: int
    finish_reason: str | None
    structured_mode: str


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_calls_left: int
    tokens_left: int
    # Wall-clock epoch seconds, so the deadline survives a checkpoint resume.
    deadline: float


class RunError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    node: str


class ResearchState(BaseModel):
    # Identity and input
    run_id: str
    workspace_id: str
    question: str
    data_policy: DataPolicy
    filters: RunFilters = Field(default_factory=RunFilters)
    # An explicit "as of" date from the request; the query analysis may supply one too (ADR-0010).
    as_of: date | None = None
    budget: Budget
    # Analysis
    analysis: QueryAnalysis | None = None
    plan: ResearchPlan | None = None
    # Evidence
    evidence: Annotated[dict[str, EvidenceItem], merge_evidence] = Field(default_factory=dict)
    retrieval_log: Annotated[list[RetrievalCall], operator.add] = Field(default_factory=list)
    # Every version (with typed dates) of each document that evidence came from.
    documents: Annotated[dict[str, list[VersionDates]], merge_documents] = Field(default_factory=dict)
    pending_queries: list[PendingQuery] = Field(default_factory=list)
    # Loop control
    iteration: int = 0
    resynth_count: int = 0
    sufficiency: Sufficiency | None = None
    # Output
    draft: Draft | None = None
    # The closed set of ids the synthesis prompt showed; a claim may cite only these (ADR-0009).
    shown_evidence_ids: list[str] = Field(default_factory=list)
    verification: list[ClaimVerdict] | None = None
    answer: FinalAnswer | None = None
    limitations: Annotated[list[str], operator.add] = Field(default_factory=list)
    llm_calls: Annotated[list[LLMCallRecord], operator.add] = Field(default_factory=list)
    # Failure
    status: Literal["running", "completed", "failed"] = "running"
    error: RunError | None = None


# Every class the checkpointer may deserialise (LangGraph's msgpack allowlist).
STATE_MODELS: tuple[type[BaseModel], ...] = (
    ResearchState,
    RunFilters,
    PendingQuery,
    RetrievalCall,
    LLMCallRecord,
    Budget,
    RunError,
    EvidenceItem,
    PageSpan,
    SpanRef,
    DateFact,
    VersionDates,
    AsOfResolution,
    QueryAnalysis,
    TemporalScope,
    ResearchPlan,
    SubQuestion,
    Sufficiency,
    Coverage,
    Reformulation,
    QueryRewrite,
    Draft,
    DraftClaim,
    DraftQuote,
    ClaimVerdict,
    Citation,
    FinalAnswer,
)
