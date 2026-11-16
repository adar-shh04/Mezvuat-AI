"""Claim-level semantic support: NLI of each claim against its cited span (ADR-0009, verification step 2).

This module is pure logic. The model is a protocol, implemented by `mezvuat.ml.nli` in production and by fakes in
tests. The premise for a claim is the verified quote widened to whole sentences, plus a sentence on each side, so a
quote cut mid-sentence still gives the model its full context. The hypothesis is the claim text.

Labels: `supported` (entailment), `contradicted` (contradiction) or `not_enough_info` (neither threshold reached).
A claim is contradicted when any of its cited spans contradicts it, supported when any span (or, for a claim citing
several spans, their concatenation) entails it and none contradicts it, and otherwise not enough info.

Thresholds come from calibration on owner-labelled pairs (`mezvuat.evaluation.nli_calibration`). Until that has
run, `NliThresholds()` is a provisional rule, and everything downstream says the thresholds are uncalibrated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from mezvuat.citations.checks import SpanRef
from mezvuat.citations.registry import EvidenceItem

SupportLabel = Literal["supported", "contradicted", "not_enough_info"]
SUPPORT_LABELS: tuple[SupportLabel, ...] = ("supported", "contradicted", "not_enough_info")

# Premises longer than this are cut around the quote: the model reads at most 512 tokens of premise and claim.
MAX_PREMISE_CHARS = 1500
# Abbreviations whose full stop does not end a sentence. Lower-cased, without the stop.
_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "etc", "fig", "figs", "eq", "eqs", "vs", "cf", "sec", "no", "approx", "resp", "tab", "al"}
)
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+|\n\s*\n")


class NliScores(BaseModel):
    """Class probabilities from an NLI model; they sum to 1."""

    model_config = ConfigDict(frozen=True)

    entailment: float
    contradiction: float
    neutral: float


class NliModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliScores]:
        """Score (premise, hypothesis) pairs, in order."""
        ...


PROVISIONAL_SOURCE = "provisional: a label needs at least half of the probability mass; not calibrated"


class NliThresholds(BaseModel):
    """τ_e and τ_c (ADR-0009). The defaults are a provisional rule, not a calibrated value."""

    model_config = ConfigDict(frozen=True)

    entailment: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction: float = Field(default=0.5, ge=0.0, le=1.0)
    calibrated: bool = False
    source: str = PROVISIONAL_SOURCE

    def label(self, scores: NliScores) -> SupportLabel:
        candidates: list[tuple[float, SupportLabel]] = []
        if scores.entailment >= self.entailment:
            candidates.append((scores.entailment, "supported"))
        if scores.contradiction >= self.contradiction:
            candidates.append((scores.contradiction, "contradicted"))
        if not candidates:
            return "not_enough_info"
        # With low thresholds both can pass; the more probable label wins, entailment on a tie.
        return max(candidates, key=lambda c: (c[0], c[1] == "supported"))[1]


def load_thresholds(path: Path | None) -> NliThresholds:
    """Thresholds written by `mezvuat-dataset nli-calibrate`, or the provisional rule when there is no file."""
    if path is None or not path.exists():
        return NliThresholds()
    data = json.loads(path.read_text(encoding="utf-8"))
    return NliThresholds(
        entailment=float(data["entailment"]),
        contradiction=float(data["contradiction"]),
        calibrated=bool(data.get("calibrated", False)),
        source=str(data.get("source", path.name)),
    )


def _is_boundary(text: str, match: re.Match[str]) -> bool:
    if text[match.start()] != ".":
        return True
    word = re.search(r"([A-Za-z.]+)$", text[max(0, match.start() - 12) : match.start()])
    if word is None:
        return True
    token = word.group(1).lower().lstrip(".")
    # A single letter (an initial, or "Fig. 3a."-style labels) or a known abbreviation does not end a sentence.
    return not (len(token) == 1 or token in _ABBREVIATIONS)


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Sentence spans of `text`, by a punctuation heuristic that skips common abbreviations in papers."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if not _is_boundary(text, match):
            continue
        end = match.end()
        if text[start:end].strip():
            bounds.append((start, end))
        start = end
    if text[start:].strip():
        bounds.append((start, len(text)))
    return bounds or [(0, len(text))]


def premise_window(
    text: str, start: int, end: int, *, sentences: int = 1, max_chars: int = MAX_PREMISE_CHARS
) -> tuple[str, int, int]:
    """The sentences holding `text[start:end]`, plus `sentences` on each side; whitespace is collapsed.

    Returns the premise and its offsets in `text`. Over `max_chars`, the window is cut around the span.
    """
    bounds = sentence_bounds(text)
    first = next((i for i, (a, b) in enumerate(bounds) if a <= start < b), 0)
    last = next((i for i, (a, b) in enumerate(bounds) if a < end <= b), len(bounds) - 1)
    first, last = max(0, first - sentences), min(len(bounds) - 1, max(first, last) + sentences)
    window_start, window_end = bounds[first][0], bounds[last][1]
    if window_end - window_start > max_chars:
        spare = max(0, max_chars - (end - start))
        window_start = max(window_start, start - spare // 2)
        window_end = min(window_end, window_start + max(max_chars, end - start))
    return " ".join(text[window_start:window_end].split()), window_start, window_end


class SpanSupport(BaseModel):
    """One cited span's NLI result for a claim."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    label: SupportLabel
    scores: NliScores
    premise_chars: int


class NliVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: SupportLabel
    model_id: str
    thresholds_calibrated: bool
    spans: list[SpanSupport]
    # For a claim citing several spans: their premises joined, so a claim that combines facts can still be entailed.
    combined: NliScores | None = None

    @property
    def entailment(self) -> float:
        best = max((s.scores.entailment for s in self.spans), default=0.0)
        return max(best, self.combined.entailment) if self.combined else best


def span_premise(span: SpanRef, item: EvidenceItem, *, sentences: int = 1) -> str:
    if span.char_start is None or span.char_end is None:
        return " ".join(span.quote.split())
    start, end = span.char_start - item.char_start, span.char_end - item.char_start
    return premise_window(item.text, start, end, sentences=sentences)[0]


def decide(spans: Sequence[SpanSupport], combined: NliScores | None, thresholds: NliThresholds) -> SupportLabel:
    labels = {s.label for s in spans}
    if "contradicted" in labels:
        return "contradicted"
    if "supported" in labels or (combined is not None and thresholds.label(combined) == "supported"):
        return "supported"
    return "not_enough_info"


def verify_claims(
    claims: Sequence[tuple[str, Sequence[SpanRef]]],
    registry: Mapping[str, EvidenceItem],
    model: NliModel,
    thresholds: NliThresholds,
    *,
    sentences: int = 1,
) -> list[NliVerdict]:
    """NLI for each (claim text, verified spans), in one model call. Spans must have passed the quote check."""
    pairs: list[tuple[str, str]] = []
    plan: list[tuple[list[tuple[str, int]], int | None]] = []
    for text, spans in claims:
        per_span: list[tuple[str, int]] = []
        premises: list[str] = []
        for span in {s.evidence_id: s for s in spans}.values():
            premise = span_premise(span, registry[span.evidence_id], sentences=sentences)
            premises.append(premise)
            per_span.append((span.evidence_id, len(premise)))
            pairs.append((premise, text))
        combined_index: int | None = None
        if len(premises) > 1:
            combined_index = len(pairs)
            pairs.append((" ".join(premises)[: MAX_PREMISE_CHARS * 2], text))
        plan.append((per_span, combined_index))
    scores = model.predict(pairs) if pairs else []
    if len(scores) != len(pairs):
        raise ValueError(f"the NLI model returned {len(scores)} scores for {len(pairs)} pairs")
    verdicts: list[NliVerdict] = []
    cursor = 0
    for per_span, combined_index in plan:
        results = []
        for evidence_id, chars in per_span:
            results.append(
                SpanSupport(
                    evidence_id=evidence_id,
                    label=thresholds.label(scores[cursor]),
                    scores=scores[cursor],
                    premise_chars=chars,
                )
            )
            cursor += 1
        combined = None
        if combined_index is not None:
            combined = scores[combined_index]
            cursor += 1
        verdicts.append(
            NliVerdict(
                label=decide(results, combined, thresholds),
                model_id=model.model_id,
                thresholds_calibrated=thresholds.calibrated,
                spans=results,
                combined=combined,
            )
        )
    return verdicts
