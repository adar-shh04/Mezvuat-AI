"""Evidence registry, deterministic citation checks and answer-marker validation (ADR-0009)."""

from mezvuat.citations.checks import (
    ClaimCheck,
    MarkerReport,
    QuoteInput,
    SpanRef,
    check_claim,
    check_quote,
    scan_markers,
)
from mezvuat.citations.injection import PromptLeakDetector, injection_signals
from mezvuat.citations.quotes import MAX_QUOTE_CHARS, MIN_QUOTE_CHARS, find_quote, normalize_for_match
from mezvuat.citations.registry import (
    EVIDENCE_ID,
    EvidenceItem,
    PageSpan,
    chunk_index,
    next_evidence_id,
    page_at,
)
from mezvuat.citations.support import (
    SUPPORT_LABELS,
    NliModel,
    NliScores,
    NliThresholds,
    NliVerdict,
    SpanSupport,
    SupportLabel,
    load_thresholds,
    premise_window,
    sentence_bounds,
    verify_claims,
)

__all__ = [
    "EVIDENCE_ID",
    "MAX_QUOTE_CHARS",
    "MIN_QUOTE_CHARS",
    "SUPPORT_LABELS",
    "ClaimCheck",
    "EvidenceItem",
    "MarkerReport",
    "NliModel",
    "NliScores",
    "NliThresholds",
    "NliVerdict",
    "PageSpan",
    "PromptLeakDetector",
    "QuoteInput",
    "SpanRef",
    "SpanSupport",
    "SupportLabel",
    "check_claim",
    "check_quote",
    "chunk_index",
    "find_quote",
    "injection_signals",
    "load_thresholds",
    "next_evidence_id",
    "normalize_for_match",
    "page_at",
    "premise_window",
    "scan_markers",
    "sentence_bounds",
    "verify_claims",
]
