"""Text normalisation and heading classification shared by the parsers."""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏  ﻿]")  # noqa: RUF001
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_HORIZONTAL_SPACE = re.compile(r"[ \t ]+")  # noqa: RUF001
_BLANK_LINES = re.compile(r"\n{3,}")
_NUMBERED_HEADING = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-Z]\.(?:\d{1,2}\.?)*)\s+\S")
_REFERENCES = re.compile(r"^(?:\d+\.?\s+)?(?:references|bibliography|works cited)\s*$", re.IGNORECASE)
_APPENDIX = re.compile(r"^(?:[A-Z]\.?\s+|appendix\b)", re.IGNORECASE)
_ABSTRACT = re.compile(r"^abstract\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """NFC, strip control characters, rejoin words hyphenated at line ends, squeeze whitespace."""
    cleaned = _CONTROL.sub("", unicodedata.normalize("NFC", text)).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _LINE_BREAK_HYPHEN.sub("", cleaned)
    cleaned = _HORIZONTAL_SPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return _BLANK_LINES.sub("\n\n", cleaned).strip()


def heading_level(title: str) -> int:
    """Depth implied by heading numbering: `3` -> 1, `3.2` -> 2. Unnumbered headings are top level."""
    head = title.split(maxsplit=1)[0].rstrip(".")
    return head.count(".") + 1 if _NUMBERED_HEADING.match(title) else 1


def looks_numbered(title: str) -> bool:
    return _NUMBERED_HEADING.match(title) is not None


def section_kind(title: str) -> str:
    if _REFERENCES.match(title):
        return "references"
    if _ABSTRACT.match(title):
        return "abstract"
    if title.lower().startswith("appendix") or (
        _APPENDIX.match(title) and looks_numbered(title) and title[0].isupper()
    ):
        return "appendix"
    return "body"
