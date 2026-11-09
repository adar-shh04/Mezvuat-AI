"""Structure-aware chunking with exact provenance (ADR-0003).

A chunk never crosses a section boundary, and `text[char_start:char_end]` is always exactly the chunk text.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from mezvuat.ingestion.parsers.base import PageSpan, ParsedDocument, Section

CHUNKER_VERSION = "structure-v1"
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
    def split(self, text: str, max_tokens: int) -> list[tuple[int, int]]: ...


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 350
    overlap_tokens: int = 50
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("need 0 <= overlap < target <= max")


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    section_kind: str
    token_count: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int


def chunk_id_for(document_version_id: str, chunker_version: str, ordinal: int) -> str:
    return hashlib.sha256(f"{document_version_id}:{chunker_version}:{ordinal}".encode()).hexdigest()


def _spans(text: str, pattern: re.Pattern[str], start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(text, start, end):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _units_for_span(text: str, start: int, end: int, counter: TokenCounter, config: ChunkingConfig) -> list[_Unit]:
    """Split a span into units that each fit: paragraphs, then sentences, then raw token windows."""
    units: list[_Unit] = []
    for para_start, para_end in _spans(text, _PARAGRAPH_BREAK, start, end):
        trimmed = _trim(text, para_start, para_end)
        if trimmed is None:
            continue
        para_tokens = counter.count(text[trimmed[0] : trimmed[1]])
        if para_tokens <= config.target_tokens:
            units.append(_Unit(trimmed[0], trimmed[1], para_tokens))
            continue
        for sent_start, sent_end in _spans(text, _SENTENCE_END, trimmed[0], trimmed[1]):
            sentence = _trim(text, sent_start, sent_end)
            if sentence is None:
                continue
            sent_tokens = counter.count(text[sentence[0] : sentence[1]])
            if sent_tokens <= config.target_tokens:
                units.append(_Unit(sentence[0], sentence[1], sent_tokens))
                continue
            for window_start, window_end in counter.split(text[sentence[0] : sentence[1]], config.target_tokens):
                window = _trim(text, sentence[0] + window_start, sentence[0] + window_end)
                if window is not None:
                    units.append(_Unit(window[0], window[1], counter.count(text[window[0] : window[1]])))
    return units


def _pack(units: list[_Unit], config: ChunkingConfig) -> list[list[_Unit]]:
    """Greedy packing up to the target size, with overlap taken from whole trailing units."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    tokens = 0
    for unit in units:
        if current and tokens + unit.tokens > config.target_tokens:
            groups.append(current)
            overlap: list[_Unit] = []
            overlap_tokens = 0
            for previous in reversed(current):
                if overlap_tokens + previous.tokens > config.overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_tokens += previous.tokens
            # Overlap must not push the next chunk past the target, or the same unit repeats forever.
            if overlap_tokens + unit.tokens > config.target_tokens:
                overlap, overlap_tokens = [], 0
            current, tokens = overlap, overlap_tokens
        current.append(unit)
        tokens += unit.tokens
    if current and not (groups and all(u in groups[-1] for u in current)):
        groups.append(current)
    return groups


def _page_lookup(pages: list[PageSpan]) -> tuple[list[int], list[int]]:
    return [p.char_start for p in pages], [p.page_number for p in pages]


def _page_at(offset: int, starts: list[int], numbers: list[int]) -> int | None:
    if not starts:
        return None
    index = bisect.bisect_right(starts, offset) - 1
    return numbers[max(index, 0)]


def chunk_document(
    parsed: ParsedDocument, counter: TokenCounter, config: ChunkingConfig | None = None
) -> list[ChunkDraft]:
    config = config or ChunkingConfig()
    text = parsed.text
    starts, numbers = _page_lookup(parsed.pages)
    drafts: list[ChunkDraft] = []
    for section in _ordered_sections(parsed.sections):
        units = _units_for_span(text, section.char_start, section.char_end, counter, config)
        for group in _pack(units, config):
            start, end = group[0].start, group[-1].end
            body = text[start:end]
            if section.path and " ".join(body.split()) == section.path[-1]:
                continue  # a heading alone is not evidence; its title is carried in the children section paths
            tokens = counter.count(body)
            drafts.append(
                ChunkDraft(
                    ordinal=len(drafts),
                    char_start=start,
                    char_end=end,
                    page_start=_page_at(start, starts, numbers),
                    page_end=_page_at(end - 1, starts, numbers),
                    section_path=section.path,
                    section_kind=section.kind,
                    token_count=tokens,
                    text=body,
                    text_sha256=hashlib.sha256(body.encode()).hexdigest(),
                )
            )
    return drafts


def _ordered_sections(sections: list[Section]) -> list[Section]:
    return sorted(sections, key=lambda s: s.char_start)


def embedding_text(title: str | None, section_path: tuple[str, ...], text: str) -> str:
    """Context prefix used for embedding only. It is never stored as the chunk text and never cited."""
    prefix = " > ".join(part for part in (title, *section_path) if part)
    return f"{prefix}\n{text}" if prefix else text
