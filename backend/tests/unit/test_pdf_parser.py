from __future__ import annotations

import itertools

import pymupdf
import pytest

from mezvuat.core.errors import ParseFailedError
from mezvuat.ingestion.parsers.pdf import PdfParser


def single_column_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 60), "A Study of Hybrid Retrieval", fontsize=22, fontname="hebo")
    page.insert_text((72, 110), "1 Introduction", fontsize=14, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(72, 130, 520, 260), "Lexical search finds exact terminology in papers. " * 3, fontsize=10
    )
    page.insert_text((72, 300), "1.1 Motivation", fontsize=12, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(72, 320, 520, 420),
        "Dense retrievers capture paraphrases of the query. " * 3,
        fontsize=10,
    )
    page = document.new_page()
    page.insert_text((72, 72), "2 Method", fontsize=14, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(72, 90, 520, 200), "We fuse rankings by reciprocal rank on page two. " * 3, fontsize=10
    )
    page.insert_text((72, 240), "References", fontsize=14, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 260, 520, 360), "[1] Cormack et al. Fusion. SIGIR 2009.", fontsize=10)
    data: bytes = document.tobytes()
    document.close()
    return data


def two_column_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    left = "Alpha column sentence one continues here. " * 6
    right = "Omega column sentence two continues there. " * 6
    page.insert_textbox(pymupdf.Rect(40, 60, 290, 600), left, fontsize=10)
    page.insert_textbox(pymupdf.Rect(310, 60, 570, 600), right, fontsize=10)
    data: bytes = document.tobytes()
    document.close()
    return data


def test_pages_are_exact_character_ranges() -> None:
    parsed = PdfParser().parse(single_column_pdf())
    assert [p.page_number for p in parsed.pages] == [1, 2]
    assert "Lexical search" in parsed.text[parsed.pages[0].char_start : parsed.pages[0].char_end]
    assert "reciprocal rank" in parsed.text[parsed.pages[1].char_start : parsed.pages[1].char_end]
    assert "reciprocal rank" not in parsed.text[parsed.pages[0].char_start : parsed.pages[0].char_end]


def test_section_tree_from_font_and_numbering() -> None:
    parsed = PdfParser().parse(single_column_pdf())
    paths = [s.path for s in parsed.sections]
    assert ("1 Introduction",) in paths
    assert ("1 Introduction", "1.1 Motivation") in paths
    assert ("2 Method",) in paths
    assert ("References",) in paths
    references = next(s for s in parsed.sections if s.path == ("References",))
    assert references.kind == "references"
    assert parsed.text[references.char_start :].startswith("References")


def test_sections_cover_the_text_without_overlap() -> None:
    parsed = PdfParser().parse(single_column_pdf())
    ordered = sorted(parsed.sections, key=lambda s: s.char_start)
    assert ordered[0].char_start == 0
    assert ordered[-1].char_end == len(parsed.text)
    for current, following in itertools.pairwise(ordered):
        assert current.char_end == following.char_start


def test_title_is_the_largest_text_on_page_one() -> None:
    assert PdfParser().parse(single_column_pdf()).title == "A Study of Hybrid Retrieval"


def test_two_columns_are_not_interleaved() -> None:
    parsed = PdfParser().parse(two_column_pdf())
    assert parsed.text.index("Alpha") < parsed.text.index("Omega")
    assert "Omega" not in parsed.text[: parsed.text.rindex("Alpha")]


def test_rotated_margin_stamp_is_dropped() -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 500, 300), "Body text of the paper.", fontsize=10)
    page.insert_text((20, 500), "arXiv:2401.00001v1 [cs.IR]", fontsize=20, rotate=90)
    parsed = PdfParser().parse(document.tobytes())
    assert "arXiv:" not in parsed.text
    assert "Body text" in parsed.text


def test_corrupt_bytes_are_a_parse_failure() -> None:
    with pytest.raises(ParseFailedError):
        PdfParser().parse(b"%PDF-1.7 not actually a pdf")


def test_blank_pdf_yields_no_text() -> None:
    document = pymupdf.open()
    document.new_page()
    assert PdfParser().parse(document.tobytes()).text == ""
