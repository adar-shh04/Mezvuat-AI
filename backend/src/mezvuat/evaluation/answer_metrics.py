"""Mezvuat's own answer-level metrics (ADR-0012, M6). Deterministic: no LLM, no network.

Each function reads one finished research run (`ResearchState`) and, where needed, its gold item. A metric that does
not apply to an item (no citations, no gold chunks, no expected versions) is absent rather than scored 0 or 1, so
averages are over the items it applies to, and `n` says how many.

Gold item fields (JSONL, one object per item):
- `item_id`, `question` (required);
- `reference_answer` (RAGAS context precision and recall);
- `expected_behavior`: `answer` or `abstain` (`cannot_determine` is read as `abstain`);
- `gold_chunk_ids` (citation precision and recall), `expected_versions` (temporal accuracy), `as_of`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from mezvuat.citations import QuoteInput, check_quote
from mezvuat.orchestration.state import ResearchState

METRIC_VERSION = "mezvuat-answer-metrics-1"
_MARKER = re.compile(r"\[(E\d+)\]")
ABSTAIN = frozenset({"abstain", "cannot_determine"})


def abstained(state: ResearchState) -> bool:
    answer = state.answer
    return answer is None or not any(claim.status == "supported" for claim in answer.claims)


def citation_validity(state: ResearchState) -> float | None:
    """Share of the answer's citation markers that resolve to a citation whose quote passes the span check against
    the evidence registry. ADR-0012 requires 1.0."""
    answer = state.answer
    if answer is None:
        return None
    markers = _MARKER.findall(answer.markdown)
    if not markers and not answer.citations:
        return None
    registry = state.evidence
    by_marker: dict[str, list[Any]] = {}
    for citation in answer.citations:
        by_marker.setdefault(citation.evidence_id, []).append(citation)
    checks: list[bool] = []
    for marker in markers or [c.evidence_id for c in answer.citations]:
        citations = by_marker.get(marker, [])
        ok = bool(citations) and all(
            check_quote(QuoteInput(evidence_id=c.evidence_id, text=c.quote), {c.evidence_id}, registry).status == "ok"
            for c in citations
        )
        checks.append(ok)
    return sum(checks) / len(checks)


def citation_support_rate(state: ResearchState) -> float | None:
    """Share of cited claims whose support was confirmed semantically (NLI-entailed or judge-verified)."""
    answer = state.answer
    if answer is None:
        return None
    cited = [c for c in answer.claims if c.evidence_ids]
    if not cited:
        return None
    return sum(1 for c in cited if c.status == "supported" and c.method in ("nli", "llm_judge")) / len(cited)


def citation_precision_recall(state: ResearchState, gold_chunk_ids: list[str] | None) -> dict[str, float]:
    answer = state.answer
    if not gold_chunk_ids or answer is None:
        return {}
    gold = set(gold_chunk_ids)
    cited = {c.chunk_id for c in answer.citations}
    out = {"citation_recall": len(cited & gold) / len(gold)}
    if cited:
        out["citation_precision"] = len(cited & gold) / len(cited)
    return out


def claim_outcomes(state: ResearchState) -> dict[str, float]:
    verdicts = state.verification or []
    if not verdicts:
        return {}
    unsupported = [v for v in verdicts if v.status != "supported"]
    return {"unsupported_claim_rate": len(unsupported) / len(verdicts), "claims_dropped": float(len(unsupported))}


def expects_abstention(item: Mapping[str, Any]) -> bool | None:
    behavior = item.get("expected_behavior")
    return None if behavior is None else behavior in ABSTAIN


def abstention_correct(state: ResearchState, item: Mapping[str, Any]) -> float | None:
    expected = expects_abstention(item)
    return None if expected is None else float(abstained(state) == expected)


def temporal_accuracy(state: ResearchState, item: Mapping[str, Any]) -> float | None:
    """The temporal set (ADR-0010): the right version cited, or a correct abstention."""
    expected_versions = item.get("expected_versions")
    expected_abstain = expects_abstention(item)
    if not expected_versions and not expected_abstain:
        return None
    if expected_abstain:
        return float(abstained(state))
    answer = state.answer
    if answer is None or abstained(state):
        return 0.0
    cited = {c.version_label for c in answer.citations}
    return float(bool(cited) and cited <= set(expected_versions or []))


def cost(state: ResearchState) -> dict[str, float]:
    calls = state.llm_calls
    out = {
        "llm_calls": float(len(calls)),
        "prompt_tokens": float(sum(c.prompt_tokens or 0 for c in calls)),
        "completion_tokens": float(sum(c.completion_tokens or 0 for c in calls)),
        "llm_latency_ms": float(sum(c.latency_ms for c in calls)),
    }
    estimates = [c.cost_estimated_usd for c in calls if c.cost_estimated_usd is not None]
    if estimates:
        # ModelMesh's estimate, never billing data; the direct backends report none.
        out["cost_estimated_usd"] = float(sum(estimates))
    return out


def answer_metrics(state: ResearchState, item: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {"completed": float(state.status == "completed")}
    for name, value in (
        ("citation_validity", citation_validity(state)),
        ("citation_support_rate", citation_support_rate(state)),
        ("abstention_correct", abstention_correct(state, item)),
        ("temporal_accuracy", temporal_accuracy(state, item)),
    ):
        if value is not None:
            out[name] = value
    out.update(citation_precision_recall(state, item.get("gold_chunk_ids")))
    out.update(claim_outcomes(state))
    out.update(cost(state))
    return out
