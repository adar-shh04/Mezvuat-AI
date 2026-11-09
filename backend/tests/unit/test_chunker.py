from __future__ import annotations

import re

import pytest

from mezvuat.ingestion.chunker import ChunkingConfig, chunk_document, chunk_id_for, embedding_text
from mezvuat.ingestion.parsers.base import PageSpan, ParsedDocument, Section


class WordCounter:
    """Whitespace tokens: deterministic and needs no model download."""

    def count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, max_tokens: int) -> list[tuple[int, int]]:
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        return [(spans[i][0], spans[min(i + max_tokens, len(spans)) - 1][1]) for i in range(0, len(spans), max_tokens)]


CONFIG = ChunkingConfig(target_tokens=20, overlap_tokens=5, max_tokens=30)


def sentence(i: int, words: int = 8) -> str:
    return " ".join(f"Word{i}x{j}" for j in range(words)) + "."


def doc(text: str, sections: list[Section] | None = None, pages: list[PageSpan] | None = None) -> ParsedDocument:
    return ParsedDocument(
        text=text,
        pages=pages or [],
        sections=sections or [Section(path=(), char_start=0, char_end=len(text))],
        extractor="test",
    )


def test_offsets_reproduce_chunk_text_exactly() -> None:
    text = "\n\n".join(" ".join(sentence(i * 10 + j) for j in range(4)) for i in range(5))
    drafts = chunk_document(doc(text), WordCounter(), CONFIG)
    assert len(drafts) > 1
    for draft in drafts:
        assert text[draft.char_start : draft.char_end] == draft.text


def test_chunks_never_exceed_the_target_except_single_units() -> None:
    text = " ".join(sentence(i) for i in range(30))
    for draft in chunk_document(doc(text), WordCounter(), CONFIG):
        assert draft.token_count <= CONFIG.target_tokens


def test_oversized_sentence_is_split_by_token_windows() -> None:
    text = " ".join(f"w{i}" for i in range(95))
    drafts = chunk_document(doc(text), WordCounter(), CONFIG)
    assert all(d.token_count <= CONFIG.target_tokens for d in drafts)
    assert drafts[0].text.startswith("w0") and drafts[-1].text.endswith("w94")


def test_chunks_never_cross_section_boundaries() -> None:
    first = " ".join(sentence(i) for i in range(10))
    second = " ".join(sentence(i + 100) for i in range(10))
    text = f"{first}\n\n{second}"
    boundary = len(first) + 2
    sections = [
        Section(path=("1 Intro",), char_start=0, char_end=boundary),
        Section(path=("2 Method",), char_start=boundary, char_end=len(text)),
    ]
    drafts = chunk_document(doc(text, sections=sections), WordCounter(), CONFIG)
    for draft in drafts:
        if draft.section_path == ("1 Intro",):
            assert draft.char_end <= boundary
        else:
            assert draft.char_start >= boundary
    assert {d.section_path for d in drafts} == {("1 Intro",), ("2 Method",)}


def test_overlap_repeats_trailing_sentences() -> None:
    text = " ".join(sentence(i, words=4) for i in range(12))
    drafts = chunk_document(doc(text), WordCounter(), ChunkingConfig(target_tokens=20, overlap_tokens=8, max_tokens=30))
    assert drafts[1].char_start < drafts[0].char_end


def test_page_numbers_come_from_character_offsets() -> None:
    page_one = " ".join(sentence(i) for i in range(3))
    page_two = " ".join(sentence(i + 50) for i in range(3))
    text = f"{page_one}\n\n{page_two}"
    pages = [
        PageSpan(page_number=1, char_start=0, char_end=len(page_one)),
        PageSpan(page_number=2, char_start=len(page_one) + 2, char_end=len(text)),
    ]
    drafts = chunk_document(doc(text, pages=pages), WordCounter(), CONFIG)
    assert drafts[0].page_start == 1
    assert drafts[-1].page_end == 2


def test_section_kind_is_carried_to_chunks() -> None:
    text = sentence(1)
    sections = [Section(path=("References",), char_start=0, char_end=len(text), kind="references")]
    assert chunk_document(doc(text, sections=sections), WordCounter(), CONFIG)[0].section_kind == "references"


def test_a_section_holding_only_its_heading_produces_no_chunk() -> None:
    text = "3\nFactor Analysis\n\n3.1 Details\n\nThe factors are described in full here."
    boundary = text.index("3.1")
    sections = [
        Section(path=("3 Factor Analysis",), char_start=0, char_end=boundary),
        Section(path=("3 Factor Analysis", "3.1 Details"), char_start=boundary, char_end=len(text)),
    ]
    drafts = chunk_document(doc(text, sections=sections), WordCounter(), CONFIG)
    assert [d.section_path for d in drafts] == [("3 Factor Analysis", "3.1 Details")]


def test_chunk_ids_are_deterministic_and_distinct() -> None:
    assert chunk_id_for("ver_1", "v1", 0) == chunk_id_for("ver_1", "v1", 0)
    assert chunk_id_for("ver_1", "v1", 0) != chunk_id_for("ver_1", "v1", 1)


def test_embedding_prefix_is_separate_from_chunk_text() -> None:
    assert embedding_text("Paper", ("3 Method",), "body") == "Paper > 3 Method\nbody"
    assert embedding_text(None, (), "body") == "body"


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=10, overlap_tokens=10, max_tokens=20)
