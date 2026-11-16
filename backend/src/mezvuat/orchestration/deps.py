"""What the research graph needs from the outside, as protocols. Production adapters live in the API layer."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from mezvuat.citations import NliModel, NliThresholds, PageSpan
from mezvuat.inference import LLMClient
from mezvuat.orchestration.state import LLMCallRecord, RunFilters
from mezvuat.retrieval.types import RankedEvidence
from mezvuat.temporal import VersionDates


class EvidenceSearch(Protocol):
    def search(self, workspace_id: str, query: str, k: int, filters: RunFilters) -> list[RankedEvidence]: ...


class PageMapSource(Protocol):
    """Page character ranges per document version, in the version's normalised text."""

    def page_spans(self, workspace_id: str, version_ids: Sequence[str]) -> dict[str, list[PageSpan]]: ...


class TemporalSource(Protocol):
    """Every version of each document, with its typed dates (ADR-0010). Missing dates are simply absent."""

    def versions(self, workspace_id: str, document_ids: Sequence[str]) -> dict[str, list[VersionDates]]: ...


class CallRecorder(Protocol):
    """Persists `llm_call` rows as they happen, so a crashed or failed run still accounts for every request."""

    def record(self, records: Sequence[LLMCallRecord]) -> None: ...


class NoTemporalData:
    def versions(self, workspace_id: str, document_ids: Sequence[str]) -> dict[str, list[VersionDates]]:
        return {}


class NoPageMap:
    def page_spans(self, workspace_id: str, version_ids: Sequence[str]) -> dict[str, list[PageSpan]]:
        return {}


@dataclass
class WorkflowDeps:
    llm: LLMClient
    search: EvidenceSearch
    pages: PageMapSource = field(default_factory=NoPageMap)
    temporal: TemporalSource = field(default_factory=NoTemporalData)
    recorder: CallRecorder | None = None
    clock: Callable[[], float] = time.time
    per_query_k: int = 8
    # Per LLM call; each call also gets at most the run's remaining deadline.
    max_call_timeout_s: float = 60.0
    # The research loop only continues with at least this much time left.
    min_loop_time_s: float = 20.0
    synthesis_evidence_chars: int = 24_000
    assess_items_per_subquestion: int = 3
    assess_item_chars: int = 800
    # Claim verification (ADR-0009). Without an NLI model, claims are checked by quote match only (the M4 layer).
    nli: NliModel | None = None
    nli_thresholds: NliThresholds = field(default_factory=NliThresholds)
    # Sentences of context on each side of a quote in the NLI premise.
    nli_premise_sentences: int = 1
    # NLI is skipped (and the answer says so) when less run time than this is left.
    nli_min_time_s: float = 10.0
    # The LLM judge for claims NLI leaves undecided. Off until its agreement with the owner's labels is measured.
    llm_judge: bool = False
    max_judge_calls: int = 3
