"""Parser contract. Offsets are absolute character positions in `ParsedDocument.text`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class PageSpan:
    page_number: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Section:
    """A consecutive, non-overlapping span of the document under one heading path."""

    path: tuple[str, ...]
    char_start: int
    char_end: int
    kind: str = "body"


@dataclass
class ParsedDocument:
    text: str
    pages: list[PageSpan]
    sections: list[Section]
    extractor: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class Parser(Protocol):
    def parse(self, raw: bytes, *, filename: str | None = None) -> ParsedDocument: ...
