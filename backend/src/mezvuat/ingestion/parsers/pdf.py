"""PDF extraction with PyMuPDF (chosen on measured reading-order quality; see ADR-0003 and docs/eval/).

Text keeps PyMuPDF's block order, which follows the content stream and so handles two-column papers. Pages are
character ranges into the normalised text. Headings come from font size and weight relative to the body text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pymupdf

from mezvuat.core.errors import ParseFailedError
from mezvuat.ingestion.normalize import heading_level, looks_numbered, normalize_text, section_kind
from mezvuat.ingestion.parsers.base import PageSpan, ParsedDocument, Section

_MAX_HEADING_CHARS = 120
_HEADING_SIZE_RATIO = 1.08
_BOLD_FLAG = 16
_UNNUMBERED_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methods",
        "methodology",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
        "appendix",
    }
)


@dataclass(frozen=True)
class _Block:
    text: str
    max_size: float
    bold_share: float
    line_count: int


def _horizontal(line: dict[str, Any]) -> bool:
    direction = line.get("dir", (1.0, 0.0))
    return abs(float(direction[0]) - 1.0) < 0.01


def _blocks_of(page: pymupdf.Page) -> list[_Block]:
    """Text blocks in content-stream order. Rotated lines (e.g. the arXiv margin stamp) are dropped."""
    blocks: list[_Block] = []
    for block in page.get_text("dict", sort=False)["blocks"]:
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        bold_chars = chars = 0
        for line in block.get("lines", []):
            if not _horizontal(line):
                continue
            spans = line.get("spans", [])
            lines.append("".join(str(span.get("text", "")) for span in spans))
            for span in spans:
                length = len(str(span.get("text", "")).strip())
                if length:
                    sizes.append(float(span.get("size", 0.0)))
                    chars += length
                    if int(span.get("flags", 0)) & _BOLD_FLAG:
                        bold_chars += length
        text = normalize_text("\n".join(lines))
        if text and not text.isdigit():
            blocks.append(_Block(text, max(sizes, default=0.0), bold_chars / chars if chars else 0.0, len(lines)))
    return blocks


def _body_size(pages: list[list[_Block]]) -> float:
    weights: Counter[float] = Counter()
    for page in pages:
        for block in page:
            weights[round(block.max_size, 1)] += len(block.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _is_heading(block: _Block, body_size: float) -> bool:
    text = block.text
    if len(text) > _MAX_HEADING_CHARS or block.line_count > 2 or not any(c.isalpha() for c in text):
        return False
    if text.endswith((".", ",", ";", ":")) and not looks_numbered(text):
        return False
    bigger = block.max_size >= body_size * _HEADING_SIZE_RATIO
    bold = block.bold_share >= 0.8
    known = text.lower().rstrip(":") in _UNNUMBERED_HEADINGS
    if looks_numbered(text):
        return bigger or bold
    return (bigger and bold) or ((bigger or bold) and known)


def _title_from_first_page(pages: list[list[_Block]]) -> str | None:
    """The largest-type text on page one, when it is clearly larger than the body."""
    first = pages[0] if pages else []
    if not first:
        return None
    largest = max(first, key=lambda b: b.max_size)
    body = _body_size(pages)
    if largest.max_size < body * 1.2 or len(largest.text) > 300:
        return None
    return " ".join(largest.text.split())


class PdfParser:
    def parse(self, raw: bytes, *, filename: str | None = None) -> ParsedDocument:
        try:
            with pymupdf.open(stream=raw, filetype="pdf") as document:
                pages_blocks = [_blocks_of(page) for page in document]
                metadata = {k: str(v) for k, v in (document.metadata or {}).items() if v}
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise ParseFailedError("PDF could not be read", code="corrupt_pdf") from exc

        body_size = _body_size(pages_blocks)
        text_parts: list[str] = []
        pages: list[PageSpan] = []
        headings: list[tuple[int, str]] = []
        cursor = 0
        for page_number, blocks in enumerate(pages_blocks, start=1):
            if not blocks:
                continue
            page_start = cursor
            for index, block in enumerate(blocks):
                if text_parts:
                    text_parts.append("\n\n")
                    cursor += 2
                if index == 0:
                    page_start = cursor
                if _is_heading(block, body_size):
                    headings.append((cursor, " ".join(block.text.split())))
                text_parts.append(block.text)
                cursor += len(block.text)
            pages.append(PageSpan(page_number=page_number, char_start=page_start, char_end=cursor))
        text = "".join(text_parts)
        return ParsedDocument(
            text=text,
            pages=pages,
            sections=_sections_from_headings(text, headings),
            extractor=f"pymupdf-{pymupdf.__version__}",
            title=_title_from_first_page(pages_blocks) or metadata.get("title") or filename,
            metadata=metadata,
        )


def _sections_from_headings(text: str, headings: list[tuple[int, str]]) -> list[Section]:
    if not text:
        return []
    if not headings:
        return [Section(path=(), char_start=0, char_end=len(text))]
    sections: list[Section] = []
    if headings[0][0] > 0:
        sections.append(Section(path=(), char_start=0, char_end=headings[0][0]))
    stack: list[tuple[int, str]] = []
    for index, (start, title) in enumerate(headings):
        level = heading_level(title)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        sections.append(
            Section(path=tuple(t for _, t in stack), char_start=start, char_end=end, kind=section_kind(title))
        )
    return sections
