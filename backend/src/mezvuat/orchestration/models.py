"""Structured node outputs (ADR-0008) and the final answer shape.

Model-facing schemas stay in the portable subset; counts and lengths are enforced here, after parsing. Count
bounds (subquestions, new queries) truncate rather than fail: they are guards, and a 5th subquestion is not a
reason to lose the run. The quote length cap fails validation, which triggers the one repair retry.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mezvuat.citations import MAX_QUOTE_CHARS, SpanRef

MAX_SUBQUESTIONS = 4
MAX_NEW_QUERIES = 2


class TemporalScope(BaseModel):
    as_of: str | None = Field(description="ISO date YYYY-MM-DD if the question asks about a point in time, else null")
    wants_latest: bool
    compare_versions: bool


class QueryAnalysis(BaseModel):
    objective: str
    subject: str
    constraints: list[str]
    expected_answer_type: Literal["factoid", "explanation", "comparison", "list", "other"]
    temporal_scope: TemporalScope
    needs_multi_doc: bool
    single_hop: bool = Field(description="true if one retrieval for the question itself is enough")


class SubQuestion(BaseModel):
    id: str
    question: str


class ResearchPlan(BaseModel):
    subquestions: list[SubQuestion]
    success_criteria: str

    @field_validator("subquestions")
    @classmethod
    def _bound(cls, value: list[SubQuestion]) -> list[SubQuestion]:
        kept = [s for s in value if s.question.strip()][:MAX_SUBQUESTIONS]
        if not kept:
            raise ValueError("at least one non-empty subquestion is required")
        # Ids are renumbered so that later nodes can rely on S1..Sn.
        return [SubQuestion(id=f"S{i}", question=s.question.strip()) for i, s in enumerate(kept, start=1)]


class Coverage(BaseModel):
    subquestion_id: str
    covered: bool
    missing: str


class Sufficiency(BaseModel):
    subquestions: list[Coverage]
    rationale: str


class QueryRewrite(BaseModel):
    subquestion_id: str
    new_queries: list[str]

    @field_validator("new_queries")
    @classmethod
    def _bound(cls, value: list[str]) -> list[str]:
        return [q.strip() for q in value if q.strip()][:MAX_NEW_QUERIES]


class Reformulation(BaseModel):
    rewrites: list[QueryRewrite]


class DraftQuote(BaseModel):
    evidence_id: str
    text: str

    @field_validator("text")
    @classmethod
    def _cap(cls, value: str) -> str:
        if len(value) > MAX_QUOTE_CHARS:
            raise ValueError(f"quotes must be at most {MAX_QUOTE_CHARS} characters; quote a shorter exact span")
        return value


class DraftClaim(BaseModel):
    id: str
    text: str
    evidence_ids: list[str]
    quotes: list[DraftQuote]
    kind: Literal["fact", "interpretation"]


class Draft(BaseModel):
    claims: list[DraftClaim]
    conflicts: list[str]
    gaps: list[str]


ClaimStatus = Literal["supported", "partially_supported", "unsupported", "contradicted", "unverifiable"]
SupportMethod = Literal["quote_match", "nli", "llm_judge", "none"]


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    text: str
    kind: Literal["fact", "interpretation"]
    status: ClaimStatus
    # How support was established. M4 has only the deterministic quote match (ADR-0009 step 1).
    method: SupportMethod
    evidence_ids: list[str]
    spans: list[SpanRef]
    problems: list[str]


class Citation(BaseModel):
    """What a `[E#]` marker in the answer resolves to: the exact verified span, on a page."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    claim_id: str
    chunk_id: str
    document_id: str
    document_version_id: str
    document_title: str | None
    version_label: str | None
    page: int | None
    char_start: int
    char_end: int
    quote: str


class FinalAnswer(BaseModel):
    markdown: str
    rendering: Literal["model", "deterministic"]
    citations: list[Citation]
    claims: list[ClaimVerdict]
    limitations: list[str]
    flags: list[str]
