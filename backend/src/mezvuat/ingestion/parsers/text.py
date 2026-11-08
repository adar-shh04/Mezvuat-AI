"""Plain text and Markdown parsing. There are no pages; locations are section plus character offsets."""

from __future__ import annotations

import re

from mezvuat.core.errors import ParseFailedError
from mezvuat.ingestion.normalize import normalize_text, section_kind
from mezvuat.ingestion.parsers.base import ParsedDocument, Section

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseFailedError("text file is not valid UTF-8", code="invalid_encoding") from exc


def build_markdown_sections(text: str) -> list[Section]:
    matches = list(_ATX_HEADING.finditer(text))
    if not matches:
        return [Section(path=(), char_start=0, char_end=len(text))]
    sections: list[Section] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(Section(path=(), char_start=0, char_end=matches[0].start()))
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            Section(
                path=tuple(t for _, t in stack),
                char_start=match.start(),
                char_end=end,
                kind=section_kind(title),
            )
        )
    return sections


class PlainTextParser:
    def parse(self, raw: bytes, *, filename: str | None = None) -> ParsedDocument:
        text = normalize_text(_decode(raw))
        if not text:
            raise ParseFailedError("document contains no text", code="empty_document")
        return ParsedDocument(
            text=text,
            pages=[],
            sections=[Section(path=(), char_start=0, char_end=len(text))],
            extractor="plaintext",
            title=filename,
        )


class MarkdownParser:
    def parse(self, raw: bytes, *, filename: str | None = None) -> ParsedDocument:
        text = normalize_text(_decode(raw))
        if not text:
            raise ParseFailedError("document contains no text", code="empty_document")
        sections = build_markdown_sections(text)
        first_heading = next((s.path[0] for s in sections if s.path), None)
        return ParsedDocument(
            text=text, pages=[], sections=sections, extractor="markdown", title=first_heading or filename
        )
