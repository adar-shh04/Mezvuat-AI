"""HTML main-text extraction with trafilatura. Markup is discarded; only text reaches the index."""

from __future__ import annotations

import trafilatura

from mezvuat.core.errors import ParseFailedError
from mezvuat.ingestion.normalize import normalize_text
from mezvuat.ingestion.parsers.base import ParsedDocument
from mezvuat.ingestion.parsers.text import build_markdown_sections


class HtmlParser:
    def parse(self, raw: bytes, *, filename: str | None = None) -> ParsedDocument:
        extracted = trafilatura.extract(
            raw,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_links=False,
            include_images=False,
            favor_precision=True,
        )
        if not extracted or not extracted.strip():
            raise ParseFailedError("no main text could be extracted from the page", code="no_text_layer")
        text = normalize_text(extracted)
        metadata = trafilatura.extract_metadata(raw)
        title = metadata.title if metadata is not None else None
        return ParsedDocument(
            text=text,
            pages=[],
            sections=build_markdown_sections(text),
            extractor=f"trafilatura-{trafilatura.__version__}",
            title=title or filename,
        )
