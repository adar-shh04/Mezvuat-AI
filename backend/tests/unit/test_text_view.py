"""The evidence viewer's text window: the page holding a citation, or a bounded window without a page map."""

from __future__ import annotations

import pytest

from mezvuat.api.text_view import WINDOW_CHARS, select_window
from mezvuat.core.errors import NotFoundError

PAGES = [{"page": 2, "start": 150, "end": 400}, {"page": 1, "start": 0, "end": 150}]


def test_offset_selects_the_page_that_holds_it() -> None:
    window = select_window(400, PAGES, page=None, at=160)
    assert (window.page, window.page_count, window.start, window.end) == (2, 2, 150, 400)
    assert select_window(400, PAGES, page=None, at=149).page == 1


def test_explicit_page_wins_over_the_offset_and_defaults_to_the_first_page() -> None:
    assert select_window(400, PAGES, page=1, at=300).page == 1
    assert select_window(400, PAGES, page=None, at=None).page == 1


def test_unknown_page_or_offset_is_not_found() -> None:
    with pytest.raises(NotFoundError):
        select_window(400, PAGES, page=3, at=None)
    with pytest.raises(NotFoundError):
        select_window(400, PAGES, page=None, at=400)


def test_without_a_page_map_a_bounded_window_surrounds_the_offset() -> None:
    length = 20_000
    window = select_window(length, None, page=None, at=10_000)
    assert window.page is None and window.page_count == 0
    assert window.start <= 10_000 < window.end and window.end - window.start == WINDOW_CHARS
    short = select_window(100, [], page=None, at=50)
    assert (short.start, short.end) == (0, 100)
    with pytest.raises(NotFoundError):
        select_window(100, None, page=2, at=None)
