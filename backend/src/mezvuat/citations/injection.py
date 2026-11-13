"""Prompt-injection signals and prompt-leak detection (ADR-0009, injection defences 3 and 5).

Flagging never removes evidence (flagging should not censor). It warns the model in the evidence header, marks the
citation for the evidence viewer, and adds a limitation. The structural defences do the real work: no tools, a
closed citation set, verbatim-quote checks and marker validation.

The patterns are heuristics for text-level instructions. Hidden-text tricks in PDFs (tiny or white fonts) need the
parser's character attributes and are not detected here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|"
            r"your|the)\b[^.\n]{0,30}\b(?:instructions?|prompts?|rules|directions|guidelines|context)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_marker",
        re.compile(r"(?:^|\n)\s*(?:system|assistant|developer)\s*:|<\|im_start\|>|\[/?INST\]|<<SYS>>", re.I),
    ),
    ("persona_change", re.compile(r"\byou are now\b|\bact as (?:an?|the) (?:different|new|unrestricted)\b", re.I)),
    (
        "prompt_extraction",
        re.compile(
            r"\b(?:reveal|print|output|repeat|show|leak|disclose)\b[^.\n]{0,30}\b(?:system prompt|your prompt|"
            r"your instructions|hidden instructions|initial prompt)\b",
            re.IGNORECASE,
        ),
    ),
    ("citation_manipulation", re.compile(r"\bcite\s+(?:source\s+|evidence\s+)?\[?E\d+\]?|\[E\d{2,}\]", re.I)),
    (
        "answer_dictation",
        re.compile(
            # Not "the final answer is ...": papers on QA write that all the time.
            r"\b(?:the correct answer (?:is|must be)|the (?:final )?answer must be|always (?:answer|respond|say)|"
            r"you must (?:say|answer|respond|output|state))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "addressed_to_model",
        re.compile(
            r"\b(?:note|message|instruction)s? (?:to|for) (?:the )?(?:ai|llm|language model|assistant|chatbot)s?\b",
            re.I,
        ),
    ),
    ("delimiter_forgery", re.compile(r"<\s*/?\s*evidence\b|boundary\s*=", re.I)),
)


def injection_signals(text: str) -> list[str]:
    """Names of the instruction-like patterns found in `text`, in a stable order."""
    return [name for name, pattern in _SIGNALS if pattern.search(text)]


_WORDS = re.compile(r"[a-z0-9]+")
SHINGLE_WORDS = 8


def _shingles(text: str, size: int = SHINGLE_WORDS) -> set[tuple[str, ...]]:
    words = _WORDS.findall(text.lower())
    return {tuple(words[i : i + size]) for i in range(max(len(words) - size + 1, 0))}


class PromptLeakDetector:
    """Flags output that reproduces 8 or more consecutive words of any of our prompts."""

    def __init__(self, prompts: Iterable[str]) -> None:
        self._shingles: set[tuple[str, ...]] = set()
        for prompt in prompts:
            self._shingles |= _shingles(prompt)

    def leaks(self, text: str) -> bool:
        return not self._shingles.isdisjoint(_shingles(text))
