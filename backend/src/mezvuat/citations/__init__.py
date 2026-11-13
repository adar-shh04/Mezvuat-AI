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
from mezvuat.citations.quotes import MAX_QUOTE_CHARS, MIN_QUOTE_CHARS, find_quote, normalize_for_match
from mezvuat.citations.registry import (
    EVIDENCE_ID,
    EvidenceItem,
    PageSpan,
    chunk_index,
    next_evidence_id,
    page_at,
)

__all__ = [
    "EVIDENCE_ID",
    "MAX_QUOTE_CHARS",
    "MIN_QUOTE_CHARS",
    "ClaimCheck",
    "EvidenceItem",
    "MarkerReport",
    "PageSpan",
    "QuoteInput",
    "SpanRef",
    "check_claim",
    "check_quote",
    "chunk_index",
    "find_quote",
    "next_evidence_id",
    "normalize_for_match",
    "page_at",
    "scan_markers",
]
