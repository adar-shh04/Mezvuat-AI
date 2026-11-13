"""Deterministic quote matching (ADR-0009, verification step 1).

A quote matches when its normalised form is a substring of the normalised chunk text. Normalisation keeps case and:
- applies NFKC per character (ligatures such as `fi` become `fi`);
- rejoins words hyphenated at a line break (`retrie-\\nval` becomes `retrieval`);
- collapses every whitespace run to one space;
- folds typographic quotes and dashes to their ASCII forms, because models rarely reproduce them.
The match is mapped back to offsets in the original chunk text, so the citation points at the exact span.
Fuzzy matching for PDF-extraction artefacts (rapidfuzz, ratio >= 0.95) is an M5 addition.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 300
# Typographic single quotes, double quotes and dashes, by code point, folded to ASCII.
_FOLD = str.maketrans(
    {chr(c): "'" for c in (0x2018, 0x2019, 0x201A, 0x201B)}
    | {chr(c): '"' for c in (0x201C, 0x201D, 0x201E)}
    | {chr(c): "-" for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)}
)
_WRAPPING = "\"'" + "".join(chr(c) for c in (0x201C, 0x201D, 0x2018, 0x2019))


@dataclass(frozen=True)
class Normalized:
    text: str
    # origin[i] is the index in the source string of normalised character i.
    origin: list[int]


def normalize_for_match(source: str) -> Normalized:
    out: list[str] = []
    origin: list[int] = []
    i = 0
    n = len(source)
    while i < n:
        char = source[i]
        if char == "-":
            # Hyphen, optional spaces, a newline, optional spaces: a word broken at a line end.
            j = i + 1
            while j < n and source[j] in " \t":
                j += 1
            if j < n and source[j] == "\n":
                j += 1
                while j < n and source[j] in " \t":
                    j += 1
                if out and out[-1].isalpha() and j < n and source[j].isalpha():
                    i = j
                    continue
        if char.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                origin.append(i)
            i += 1
            continue
        for piece in unicodedata.normalize("NFKC", char).translate(_FOLD):
            if piece.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
                    origin.append(i)
                continue
            out.append(piece)
            origin.append(i)
        i += 1
    while out and out[-1] == " ":
        out.pop()
        origin.pop()
    return Normalized("".join(out), origin)


def clean_quote(quote: str) -> str:
    text = quote.strip()
    while len(text) >= 2 and text[0] in _WRAPPING and text[-1] in _WRAPPING:
        text = text[1:-1].strip()
    return text


def find_quote(chunk_text: str, quote: str) -> tuple[int, int] | None:
    """Offsets `(start, end)` of the quote within `chunk_text`, or None if it is not there verbatim."""
    needle = normalize_for_match(clean_quote(quote)).text
    if not needle:
        return None
    haystack = normalize_for_match(chunk_text)
    position = haystack.text.find(needle)
    if position < 0:
        return None
    start = haystack.origin[position]
    end = haystack.origin[position + len(needle) - 1] + 1
    return start, end


def normalized_length(quote: str) -> int:
    return len(normalize_for_match(clean_quote(quote)).text)
