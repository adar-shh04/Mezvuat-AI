"""The run-local evidence registry (ADR-0009). Only ids in the registry can be cited."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

EVIDENCE_ID = re.compile(r"^E[1-9]\d*$")


class PageSpan(BaseModel):
    """A page's character range in the version's normalised text."""

    model_config = ConfigDict(frozen=True)

    page: int
    start: int
    end: int


class EvidenceItem(BaseModel):
    """One retrieved chunk shown to the model, with everything needed to cite it to a page and span."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    document_id: str
    document_version_id: str
    document_title: str | None
    version_label: str | None
    is_latest_version: bool | None
    text: str
    # Offsets of `text` in the version's normalised text; `text == normalized_text[char_start:char_end]`.
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    section_kind: str
    page_spans: list[PageSpan]
    retrieval_round: int
    subquestion_id: str
    query: str
    rank: int
    score: float


def next_evidence_id(registry: Mapping[str, EvidenceItem]) -> str:
    return f"E{len(registry) + 1}"


def chunk_index(registry: Mapping[str, EvidenceItem]) -> dict[str, str]:
    return {item.chunk_id: evidence_id for evidence_id, item in registry.items()}


def page_at(item: EvidenceItem, absolute_offset: int) -> int | None:
    """The page holding an absolute offset. Without a page map, only a single-page chunk can answer."""
    for span in item.page_spans:
        if span.start <= absolute_offset < span.end:
            return span.page
    if item.page_start is not None and item.page_start == item.page_end:
        return item.page_start
    return None
