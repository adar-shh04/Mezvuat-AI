from __future__ import annotations

import pytest

from mezvuat.core.errors import ParseFailedError, PayloadTooLargeError, UnsupportedMediaError
from mezvuat.ingestion.normalize import heading_level, normalize_text, section_kind
from mezvuat.ingestion.parsers.text import MarkdownParser, PlainTextParser
from mezvuat.ingestion.validation import detect_source_type, validate_size


def test_normalize_rejoins_line_end_hyphenation_and_strips_controls() -> None:
    assert normalize_text("re-\ntrieval sys\x00tems") == "retrieval systems"
    # A real compound across a line keeps its hyphen only when not lowercase-joined.
    assert normalize_text("Pre-\nTrain") == "Pre-\nTrain"


def test_normalize_squeezes_whitespace_but_keeps_paragraphs() -> None:
    assert normalize_text("a   b\n\n\n\nc") == "a b\n\nc"


@pytest.mark.parametrize(
    ("title", "level"), [("3 Method", 1), ("3.2 Training", 2), ("3.2.1 Loss", 3), ("Conclusion", 1)]
)
def test_heading_levels(title: str, level: int) -> None:
    assert heading_level(title) == level


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("References", "references"),
        ("7 Bibliography", "references"),
        ("Abstract", "abstract"),
        ("Appendix B", "appendix"),
        ("3 Method", "body"),
    ],
)
def test_section_kinds(title: str, kind: str) -> None:
    assert section_kind(title) == kind


def test_markdown_sections_track_heading_paths() -> None:
    parsed = MarkdownParser().parse(b"# Title\n\nintro\n\n## Part A\n\ntext a\n\n## Part B\n\ntext b\n")
    assert [s.path for s in parsed.sections] == [("Title",), ("Title", "Part A"), ("Title", "Part B")]
    assert parsed.title == "Title"
    for section in parsed.sections:
        assert parsed.text[section.char_start :].startswith("#")
    assert parsed.sections[-1].char_end == len(parsed.text)


def test_markdown_without_headings_is_one_section() -> None:
    parsed = MarkdownParser().parse(b"just some text")
    assert [s.path for s in parsed.sections] == [()]


def test_plain_text_rejects_empty_and_non_utf8() -> None:
    with pytest.raises(ParseFailedError):
        PlainTextParser().parse(b"   \n ")
    with pytest.raises(ParseFailedError):
        PlainTextParser().parse(b"\xff\xfe\x00bad")


def test_bytes_decide_the_type_not_the_filename() -> None:
    assert detect_source_type(b"%PDF-1.7 ...", "renamed.txt") == "pdf"
    assert detect_source_type(b"# hello", "notes.md") == "markdown"
    assert detect_source_type(b"hello", "notes.txt") == "text"
    assert detect_source_type(b"hello", None) == "text"


@pytest.mark.parametrize(
    ("raw", "name"),
    [(b"not a pdf", "a.pdf"), (b"data", "a.exe"), (b"a\x00b", "a.txt"), (b"\xff\xfe", "a.txt")],
)
def test_rejected_uploads(raw: bytes, name: str) -> None:
    with pytest.raises(UnsupportedMediaError):
        detect_source_type(raw, name)


def test_size_limits() -> None:
    with pytest.raises(ParseFailedError):
        validate_size(b"", 10)
    with pytest.raises(PayloadTooLargeError):
        validate_size(b"x" * 11, 10)
    validate_size(b"x" * 10, 10)
