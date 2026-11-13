"""Deterministic citation checks (ADR-0009) and final-answer marker validation.

These checks are necessary, not sufficient: a verbatim quote proves the cited text exists at that location, not
that it supports the claim. Semantic support (NLI, then an LLM judge with a quote re-check) is layered on in M5.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from mezvuat.citations.quotes import MAX_QUOTE_CHARS, MIN_QUOTE_CHARS, find_quote, normalized_length
from mezvuat.citations.registry import EvidenceItem, page_at

QuoteStatus = Literal["ok", "unknown_evidence", "not_cited_by_claim", "not_found", "too_short", "too_long"]

# `[E3]` and grouped forms such as `[E3, E7]`.
_MARKER_GROUP = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]")
_MARKER_ID = re.compile(r"E\d+")
# Anything else that looks like a citation: `[3]`, `[1-2]`, `[Source 2]`, `[doc 1]`, `[E-4]`, `[E3 p.2]`.
_PSEUDO_CITATION = re.compile(
    r"\[\s*(?:\d+(?:\s*[,-]\s*\d+)*|(?:source|doc|document|ref|evidence)\s*[#:_-]?\s*\d+|E[\s_#:-]+\d+|E\d+[^\]\d,;][^\]]*)\s*\]",
    re.IGNORECASE,
)


class SpanRef(BaseModel):
    """A verified quote, resolved to absolute offsets in the version text and to a page."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    quote: str
    status: QuoteStatus
    char_start: int | None = None
    char_end: int | None = None
    page: int | None = None


class ClaimCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    problems: list[str]
    spans: list[SpanRef]


class QuoteInput(BaseModel):
    evidence_id: str
    text: str


def check_quote(quote: QuoteInput, cited_ids: set[str], registry: Mapping[str, EvidenceItem]) -> SpanRef:
    item = registry.get(quote.evidence_id)
    if item is None:
        return SpanRef(evidence_id=quote.evidence_id, quote=quote.text, status="unknown_evidence")
    if quote.evidence_id not in cited_ids:
        return SpanRef(evidence_id=quote.evidence_id, quote=quote.text, status="not_cited_by_claim")
    length = normalized_length(quote.text)
    if length < MIN_QUOTE_CHARS:
        return SpanRef(evidence_id=quote.evidence_id, quote=quote.text, status="too_short")
    if length > MAX_QUOTE_CHARS:
        return SpanRef(evidence_id=quote.evidence_id, quote=quote.text, status="too_long")
    found = find_quote(item.text, quote.text)
    if found is None:
        return SpanRef(evidence_id=quote.evidence_id, quote=quote.text, status="not_found")
    start, end = item.char_start + found[0], item.char_start + found[1]
    return SpanRef(
        evidence_id=quote.evidence_id,
        quote=item.text[found[0] : found[1]],
        status="ok",
        char_start=start,
        char_end=end,
        page=page_at(item, start),
    )


def check_claim(
    evidence_ids: Iterable[str], quotes: Iterable[QuoteInput], registry: Mapping[str, EvidenceItem]
) -> ClaimCheck:
    """Every cited id must exist, every quote must be verbatim in its cited chunk, every cited id needs a quote."""
    cited = list(dict.fromkeys(evidence_ids))
    cited_set = set(cited)
    problems: list[str] = []
    if not cited:
        problems.append("claim cites no evidence")
    for evidence_id in cited:
        if evidence_id not in registry:
            problems.append(f"{evidence_id} is not in the evidence registry")
    spans = [check_quote(q, cited_set, registry) for q in quotes]
    for span in spans:
        if span.status != "ok":
            problems.append(f"quote for {span.evidence_id}: {span.status}")
    quoted = {s.evidence_id for s in spans if s.status == "ok"}
    for evidence_id in cited:
        if evidence_id in registry and evidence_id not in quoted:
            problems.append(f"{evidence_id} has no verified quote")
    return ClaimCheck(passed=not problems, problems=problems, spans=spans)


class MarkerReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    used: list[str]
    unknown: list[str]
    pseudo_citations: list[str]

    @property
    def valid(self) -> bool:
        return not self.unknown and not self.pseudo_citations


def scan_markers(markdown: str, allowed: set[str]) -> MarkerReport:
    used: list[str] = []
    for group in _MARKER_GROUP.finditer(markdown):
        used.extend(_MARKER_ID.findall(group.group(1)))
    remainder = _MARKER_GROUP.sub("", markdown)
    pseudo = [m.group(0) for m in _PSEUDO_CITATION.finditer(remainder)]
    ordered = list(dict.fromkeys(used))
    return MarkerReport(used=ordered, unknown=[m for m in ordered if m not in allowed], pseudo_citations=pseudo)
