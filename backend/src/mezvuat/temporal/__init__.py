"""Temporal reasoning over document versions (ADR-0010).

Rules:
- Dates of different kinds are never substituted for each other, and `ingested_at` is never an input here: it is
  not a `DateFact` kind, so it cannot decide "latest" or "as of".
- "As of D" picks, per document, the version with the greatest `revised` (else `published`) date <= D among the
  versions whose date precision can decide the comparison. A year-precision date in D's year, a month-precision
  date in D's month, or a missing date cannot decide it: that version is "temporally unresolved" and is excluded
  from as-of claims.
- `effective_from` / `effective_to`, when present, take precedence for applicability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

DateKind = Literal["published", "revised", "effective_from", "effective_to"]
Precision = Literal["day", "month", "year"]
_SOURCE_RANK = {"source_metadata": 0, "user": 1, "extracted_text": 2, "pdf_info": 3}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}
SOURCE_LABELS = {
    "source_metadata": "source metadata",
    "user": "entered by a user",
    "extracted_text": "from the document text",
    "pdf_info": "PDF file info",
}


class DateFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: DateKind
    value: date
    precision: Precision
    source: str
    confidence: str


class VersionDates(BaseModel):
    """One document version as temporal reasoning sees it."""

    model_config = ConfigDict(frozen=True)

    version_id: str
    version_label: str | None
    version_ordinal: int | None
    is_latest_version: bool | None
    facts: list[DateFact]


class AsOfResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    as_of: date
    applicable: str | None
    unresolved: list[str]
    not_yet_available: list[str]
    basis: DateFact | None


def _best(facts: Iterable[DateFact], kind: str) -> DateFact | None:
    candidates = [f for f in facts if f.kind == kind]
    if not candidates:
        return None
    return min(
        candidates, key=lambda f: (_SOURCE_RANK.get(f.source, 9), _CONFIDENCE_RANK.get(f.confidence, 9), f.value)
    )


def version_date(version: VersionDates) -> DateFact | None:
    """The date that orders versions: `revised`, else `published`. Never `ingested_at`."""
    return _best(version.facts, "revised") or _best(version.facts, "published")


def compare_to(fact: DateFact, day: date) -> Literal["before_or_on", "after", "undecidable"]:
    """Is the (imprecise) date on or before `day`? Only as far as its precision allows."""
    if fact.precision == "day":
        return "before_or_on" if fact.value <= day else "after"
    if fact.precision == "month":
        if (fact.value.year, fact.value.month) == (day.year, day.month):
            return "undecidable"
        return "before_or_on" if (fact.value.year, fact.value.month) < (day.year, day.month) else "after"
    if fact.value.year == day.year:
        return "undecidable"
    return "before_or_on" if fact.value.year < day.year else "after"


def _effective(version: VersionDates, day: date) -> Literal["applies", "not_applicable", "undecidable", "absent"]:
    start = _best(version.facts, "effective_from")
    end = _best(version.facts, "effective_to")
    if start is None and end is None:
        return "absent"
    if start is not None:
        position = compare_to(start, day)
        if position == "undecidable":
            return "undecidable"
        if position == "after":
            return "not_applicable"
    if end is not None:
        position = compare_to(end, day)
        if position == "undecidable":
            return "undecidable"
        # effective_to is the first day it no longer applies only at day precision; treat "on or before" as ended.
        if position == "before_or_on":
            return "not_applicable"
    return "applies"


def resolve_as_of(versions: Iterable[VersionDates], as_of: date) -> AsOfResolution:
    candidates: list[tuple[DateFact, VersionDates]] = []
    unresolved: list[str] = []
    future: list[str] = []
    effective_hits: list[VersionDates] = []
    for version in versions:
        effective = _effective(version, as_of)
        if effective == "applies":
            effective_hits.append(version)
            continue
        if effective == "not_applicable":
            future.append(version.version_id)
            continue
        if effective == "undecidable":
            unresolved.append(version.version_id)
            continue
        fact = version_date(version)
        if fact is None:
            unresolved.append(version.version_id)
            continue
        position = compare_to(fact, as_of)
        if position == "undecidable":
            unresolved.append(version.version_id)
        elif position == "after":
            future.append(version.version_id)
        else:
            candidates.append((fact, version))
    if len(effective_hits) == 1:
        hit = effective_hits[0]
        return AsOfResolution(
            as_of=as_of,
            applicable=hit.version_id,
            unresolved=unresolved,
            not_yet_available=future,
            basis=_best(hit.facts, "effective_from"),
        )
    if len(effective_hits) > 1:
        unresolved.extend(v.version_id for v in effective_hits)
    if not candidates:
        return AsOfResolution(as_of=as_of, applicable=None, unresolved=unresolved, not_yet_available=future, basis=None)
    newest = max(fact.value for fact, _ in candidates)
    top = [(fact, v) for fact, v in candidates if fact.value == newest]
    if len(top) > 1:
        # Two versions dated the same day (or the same coarse period) cannot be ordered.
        unresolved.extend(v.version_id for _, v in top)
        return AsOfResolution(as_of=as_of, applicable=None, unresolved=unresolved, not_yet_available=future, basis=None)
    fact, version = top[0]
    return AsOfResolution(
        as_of=as_of, applicable=version.version_id, unresolved=unresolved, not_yet_available=future, basis=fact
    )


def describe_dates(facts: Iterable[DateFact]) -> str:
    """Dates labelled by kind, e.g. `published 2024-05-01 (day, source metadata)`; `dates unknown` if none."""
    parts = []
    for kind in ("published", "revised", "effective_from", "effective_to"):
        fact = _best(facts, kind)
        if fact is None:
            continue
        shown = {"day": fact.value.isoformat(), "month": fact.value.strftime("%Y-%m"), "year": str(fact.value.year)}[
            fact.precision
        ]
        label = kind.replace("_", " ")
        parts.append(f"{label} {shown} ({SOURCE_LABELS.get(fact.source, fact.source)})")
    return "; ".join(parts) if parts else "dates unknown"


def latest_status(versions: Mapping[str, VersionDates]) -> Literal["known", "unknown"]:
    return "known" if any(v.is_latest_version is True for v in versions.values()) else "unknown"


__all__ = [
    "SOURCE_LABELS",
    "AsOfResolution",
    "DateFact",
    "VersionDates",
    "compare_to",
    "describe_dates",
    "latest_status",
    "resolve_as_of",
    "version_date",
]
