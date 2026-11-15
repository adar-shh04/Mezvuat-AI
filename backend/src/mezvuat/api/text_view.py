"""Which slice of a version's text the evidence viewer shows: one page, or a window when there is no page map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mezvuat.core.errors import NotFoundError

# Without a page map (TXT, Markdown, HTML), the viewer gets this many characters around the requested offset.
WINDOW_CHARS = 4000


@dataclass(frozen=True)
class TextWindow:
    page: int | None
    page_count: int
    start: int
    end: int


def select_window(
    text_length: int, pages: list[dict[str, Any]] | None, *, page: int | None, at: int | None
) -> TextWindow:
    """`page` wins over `at`. Offsets are absolute in the normalized text, as citations store them."""
    spans = sorted(pages or [], key=lambda p: int(p["page"]))
    if spans:
        if page is not None:
            chosen = next((p for p in spans if int(p["page"]) == page), None)
            if chosen is None:
                raise NotFoundError(f"page {page} not found")
        elif at is not None:
            chosen = next((p for p in spans if int(p["start"]) <= at < int(p["end"])), None)
            if chosen is None:
                raise NotFoundError("offset is outside the document text")
        else:
            chosen = spans[0]
        return TextWindow(int(chosen["page"]), len(spans), int(chosen["start"]), int(chosen["end"]))
    if page is not None and page != 1:
        raise NotFoundError("this version has no page map")
    if at is not None and not 0 <= at < max(text_length, 1):
        raise NotFoundError("offset is outside the document text")
    centre = at or 0
    start = max(0, min(centre - WINDOW_CHARS // 2, text_length - WINDOW_CHARS))
    return TextWindow(None, 0, start, min(text_length, start + WINDOW_CHARS))
