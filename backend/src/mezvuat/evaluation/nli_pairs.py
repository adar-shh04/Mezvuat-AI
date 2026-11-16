"""Claim/evidence pairs for calibrating the NLI thresholds (ADR-0009, M5 part 3).

Every pair is built from the public demo corpus without an LLM. A claim is a sentence of a passage, a light rewrite
of it, a perturbation meant to contradict it, or a sentence from elsewhere; the premise is what the verifier would
see for that sentence: the sentence with one neighbour on each side (`premise_window`). Each method has an
*expected* label by construction, which is a prior for balancing the sample, never a gold label: the owner labels
every pair without seeing it (D2 amendment of 2026-11-14).

Methods, by expected label:
- supported: `sentence_verbatim` (the sentence itself, an easy anchor) and `sentence_rewrite` (a discourse opener,
  reporting frame such as "we show that", trailing parenthetical or subordinate clause removed).
- contradicted: `perturb_number` (one number changed), `perturb_negation` (a negation added or removed),
  `perturb_antonym` (one directional word swapped, e.g. higher/lower), `perturb_swap` (two named systems swapped in
  a comparative sentence).
- neutral: `same_passage_far` (a sentence of the same passage outside the premise window), `same_paper_other_passage`
  (a sentence of another section of the same paper, against the most word-overlapping window there) and
  `other_paper` (the most word-overlapping window of a different paper).

Papers come from the `val` family split only (ADR-0007): thresholds are tuned on validation data.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mezvuat.citations.support import premise_window, sentence_bounds
from mezvuat.evaluation.domain import DomainChunk

GENERATOR_VERSION = "nli-pairs-v1"
EXPECTED = ("supported", "contradicted", "neutral")
QUOTAS: dict[str, tuple[str, int]] = {
    "sentence_verbatim": ("supported", 12),
    "sentence_rewrite": ("supported", 38),
    "perturb_number": ("contradicted", 15),
    "perturb_negation": ("contradicted", 13),
    "perturb_antonym": ("contradicted", 16),
    "perturb_swap": ("contradicted", 6),
    "same_passage_far": ("neutral", 20),
    "same_paper_other_passage": ("neutral", 18),
    "other_paper": ("neutral", 12),
}
TARGET = sum(n for _, n in QUOTAS.values())
MAX_PAIRS_PER_PAPER = 9
NEAR_DUPLICATE = 0.7

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]+")
_STOP_WORDS = (
    "the a an of and or to in on for with by from as at is are was were be been this that these those it its we our "
    "which than then can not no into over under between both each such using use used based also more most"
)
_STOP = frozenset(_STOP_WORDS.split())


@dataclass(frozen=True)
class Sentence:
    chunk: DomainChunk
    start: int
    end: int
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.chunk.chunk_id, self.start)


def _clean(text: str) -> str:
    return " ".join(text.split())


_REFERENCE = re.compile(
    r"\b(?:Proceedings|Conference on|arXiv preprint|pp\.|vol\.|Association for Computational)|"
    r"\bIn (?:Proc|[A-Z]{2,}\s*(?:\(|'|\d|,))|[A-Z]\.;|[A-Z]\., [A-Z]\.|\b(?:19|20)\d\d[a-z]?\.\s+[A-Z]"
)
_INITIALS = re.compile(r"[A-Z]\.(?:;|,\s+[A-Z]\.|\s+and\s+[A-Z])")
_YEAR = re.compile(r"\b(?:19|20)\d\d[a-z]?\b")


def looks_like_reference(text: str) -> bool:
    """A sentence from a bibliography that the chunker left in a body section."""
    return bool(_REFERENCE.search(text)) or len(_YEAR.findall(text)) >= 2


def reference_chunk(text: str) -> bool:
    """A chunk that is mostly bibliography: many author initials or many years for its length."""
    per_400 = max(1, len(text) // 400)
    return len(_INITIALS.findall(text)) >= 2 * per_400 or len(_YEAR.findall(text)) >= 3 * per_400


def good_sentence(text: str) -> bool:
    """A self-contained prose sentence: not a heading, caption fragment, reference or table row."""
    words = text.split()
    letters = sum(c.isalpha() for c in text)
    return (
        60 <= len(text) <= 280
        and 8 <= len(words) <= 50
        and text[0].isupper()
        and text.endswith(".")
        and letters >= 0.7 * len(text.replace(" ", ""))
        and not re.search(r"https?:|www\.|@|arxiv|\bdoi\b|©|\bet al\.$", text, re.IGNORECASE)
        and not re.match(r"(Table|Figure|Fig\.|Algorithm|Appendix)\s*\d", text)
        and not looks_like_reference(text)
        and not _FORMULA.search(text)
    )


# Formulas, escaped prompt text and elided examples: not claims a reader can judge from prose.
_FORMULA = re.compile(r"[=<>\\{}|^~]|\u2026|\.\.\.|[\u0370-\u03ff\u2200-\u22ff]|[\U0001d400-\U0001d7ff]")


def sentences_of(chunk: DomainChunk) -> list[Sentence]:
    found = []
    for start, end in sentence_bounds(chunk.text):
        raw = chunk.text[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        found.append(Sentence(chunk, start + lead, start + trail, _clean(raw)))
    return found


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - _STOP


def overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ----- claim transforms -----------------------------------------------------------------------------------------------

_OPENERS = re.compile(
    r"^(?:However|Moreover|Furthermore|In addition|Additionally|Specifically|In particular|Overall|Notably|"
    r"Importantly|Finally|Therefore|Thus|As a result|In contrast|Consequently|Interestingly|Similarly|Hence|"
    r"In practice|In summary|To this end|In this way),\s+(.+)$"
)
_FRAME = re.compile(
    r"^(?:(?:In this (?:paper|work|study),\s+)?(?:we|our (?:results|experiments|analysis|findings))\s+"
    r"(?:show|find|observe|demonstrate|confirm|report|note|found|showed|observed|demonstrated|reveal|revealed)|"
    r"(?:These|The|Our) (?:results|experiments|findings) (?:show|suggest|indicate|demonstrate|confirm|reveal))"
    r"\s+that\s+(.+)$"
)
_PARENTHETICAL = re.compile(r"\s*\([^()]{3,80}\)")
_TRAILING_CLAUSE = re.compile(r",\s+(?:which|while|whereas|where|although)\s+[^,;()]+(?=\.$)")


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def rewrite(sentence: str) -> str | None:
    """An entailed rewrite: drop an opener, a reporting frame, a parenthetical or a trailing subordinate clause."""
    text = sentence
    for pattern in (_OPENERS, _FRAME):
        match = pattern.match(text)
        if match:
            text = _capitalise(match.group(1))
    text = _PARENTHETICAL.sub("", text, count=1)
    text = _TRAILING_CLAUSE.sub("", text, count=1)
    text = re.sub(r"\s+([.,;])", r"\1", text)
    if not text.endswith("."):
        text += "."
    if text == sentence or len(text) < 40 or len(text.split()) < 6:
        return None
    return text


_NUMBER = re.compile(r"(?<![\w.\[-])(\d+(?:\.\d+)?)(%?)(?![\w.]?\d)(?![\w-])")
_NOT_A_QUANTITY = re.compile(r"(?:Table|Figure|Fig\.|Section|Sec\.|Eq\.|Equation|Appendix|Algorithm|Step|v)\s*$")


def perturb_number(sentence: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    for match in _NUMBER.finditer(sentence):
        value, percent = match.group(1), match.group(2)
        if _NOT_A_QUANTITY.search(sentence[: match.start()]):
            continue
        if "." not in value and 1900 <= int(value) <= 2099 and not percent:
            continue  # a year
        if "." in value:
            decimals = len(value.split(".")[1])
            number = float(value)
            new = f"{number * rng.choice((0.55, 0.7, 1.4, 1.6)):.{decimals}f}"
        else:
            number_int = int(value)
            if number_int < 2:
                continue
            new = str(rng.choice((number_int * 2, max(number_int // 2, 1), number_int + 3 + number_int // 4)))
        if percent and float(new) > 100:
            new = f"{float(value) * 0.6:.{len(value.split('.')[1]) if '.' in value else 0}f}"
        if new == value:
            continue
        changed = sentence[: match.start(1)] + new + sentence[match.end(1) :]
        return changed, {"from": value + percent, "to": new + percent}
    return None


_REMOVE_NOT = re.compile(r"\b(is|are|was|were|does|do|did|can|could|will|would|should|may|has|have)\s+not\b")
_CANNOT = re.compile(r"\bcannot\b")
_AUX = re.compile(r"\b(is|are|was|were|can|could|will|would|should|does|do|has|have|may)\s+(?!not\b)([a-z]+)")
_NEGATIVE = re.compile(r"\b(?:no|not|never|none|neither|nor|without|cannot)\b|n't\b", re.IGNORECASE)


def perturb_negation(sentence: str) -> tuple[str, dict[str, str]] | None:
    match = _REMOVE_NOT.search(sentence)
    if match:
        return sentence[: match.start()] + match.group(1) + sentence[match.end() :], {"from": "not", "to": ""}
    match = _CANNOT.search(sentence)
    if match:
        return sentence[: match.start()] + "can" + sentence[match.end() :], {"from": "cannot", "to": "can"}
    if _NEGATIVE.search(sentence):
        return None  # adding a second negation reads as nonsense, not as a contradiction
    for match in _AUX.finditer(sentence):
        if match.start() == 0 or match.group(2).endswith("ly"):
            continue
        position = match.end(1)
        return sentence[:position] + " not" + sentence[position:], {
            "from": match.group(1),
            "to": match.group(1) + " not",
        }
    return None


_ANTONYM_PAIRS = (
    ("higher", "lower"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("improves", "worsens"),
    ("improved", "worsened"),
    ("outperforms", "underperforms"),
    ("outperform", "underperform"),
    ("outperformed", "underperformed"),
    ("better", "worse"),
    ("best", "worst"),
    ("larger", "smaller"),
    ("faster", "slower"),
    ("longer", "shorter"),
    ("maximum", "minimum"),
    ("positive", "negative"),
    ("stronger", "weaker"),
    ("effective", "ineffective"),
    ("efficient", "inefficient"),
    ("superior", "inferior"),
    ("easier", "harder"),
    ("reduces", "increases"),
    ("reduced", "raised"),
    ("more", "less"),
    ("many", "few"),
    ("high", "low"),
    ("large", "small"),
    ("above", "below"),
)
_ANTONYMS = dict(_ANTONYM_PAIRS) | {b: a for a, b in _ANTONYM_PAIRS if b not in {"increases"}}
_ANTONYM_RE = re.compile(r"\b(" + "|".join(sorted(_ANTONYMS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def perturb_antonym(sentence: str) -> tuple[str, dict[str, str]] | None:
    match = _ANTONYM_RE.search(sentence)
    if match is None:
        return None
    word = match.group(1)
    replacement = _ANTONYMS[word.lower()]
    if word[0].isupper():
        replacement = _capitalise(replacement)
    return sentence[: match.start()] + replacement + sentence[match.end() :], {"from": word, "to": replacement}


_SYSTEM_NAME = re.compile(r"\b[A-Za-z]*[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9-]*\b")
_COMPARATIVE = re.compile(r"\b(outperform\w*|better|worse|higher|lower|than|over|surpass\w*|beats?)\b", re.IGNORECASE)


def perturb_swap(sentence: str) -> tuple[str, dict[str, str]] | None:
    if not _COMPARATIVE.search(sentence):
        return None
    names = [m for m in _SYSTEM_NAME.finditer(sentence) if len(m.group(0)) >= 3]
    counts = Counter(m.group(0) for m in names)
    unique = [m for m in names if counts[m.group(0)] == 1]
    if len(unique) < 2:
        return None
    first, second = unique[0], unique[1]
    swapped = (
        sentence[: first.start()]
        + second.group(0)
        + sentence[first.end() : second.start()]
        + first.group(0)
        + sentence[second.end() :]
    )
    return swapped, {"from": f"{first.group(0)} ... {second.group(0)}", "to": f"{second.group(0)} ... {first.group(0)}"}


# ----- sampling -------------------------------------------------------------------------------------------------------


@dataclass
class Builder:
    chunks: list[DomainChunk]
    seed: int

    def __post_init__(self) -> None:
        self.rng = random.Random(f"nli-pairs:{self.seed}")  # noqa: S311  (reproducible sampling)
        self.by_chunk = {c.chunk_id: sentences_of(c) for c in self.chunks}
        self.candidates = [
            s for c in self.chunks for s in self.by_chunk[c.chunk_id] if good_sentence(s.text) and len(s.text) > 0
        ]
        self.rng.shuffle(self.candidates)
        self.used: set[tuple[str, int]] = set()
        self.per_paper: Counter[str] = Counter()
        self.words = {s.key: content_words(s.text) for c in self.chunks for s in self.by_chunk[c.chunk_id]}

    def _premise(self, sentence: Sentence) -> dict[str, Any]:
        text, start, end = premise_window(sentence.chunk.text, sentence.start, sentence.end)
        return {
            "text": text,
            "chunk_id": sentence.chunk.chunk_id,
            "arxiv_id": sentence.chunk.arxiv_id,
            "version_label": sentence.chunk.version_label,
            "title": sentence.chunk.title,
            "section_path": list(sentence.chunk.section_path),
            "page_start": sentence.chunk.page_start,
            "char_start": start,
            "char_end": end,
        }

    def _row(
        self, method: str, claim: str, claim_from: Sentence, premise_at: Sentence, detail: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return {
            "claim": claim,
            "premise": self._premise(premise_at),
            "claim_source": {
                "chunk_id": claim_from.chunk.chunk_id,
                "arxiv_id": claim_from.chunk.arxiv_id,
                "sentence": claim_from.text,
                "char_start": claim_from.start,
                "char_end": claim_from.end,
            },
            "construction": {"method": method, "expected": QUOTAS[method][0], "detail": detail or {}},
        }

    def _take(self, *sentences: Sentence) -> bool:
        papers = {s.chunk.arxiv_id for s in sentences}
        if any(s.key in self.used for s in sentences) or any(self.per_paper[p] >= MAX_PAIRS_PER_PAPER for p in papers):
            return False
        self.used.update(s.key for s in sentences)
        for paper in papers:
            self.per_paper[paper] += 1
        return True

    def _same_sentence(self, method: str, make: Callable[[str], tuple[str, dict[str, str]] | None]) -> Iterable[Any]:
        for sentence in self.candidates:
            made = make(sentence.text)
            if made is None or (made[0] == sentence.text and method != "sentence_verbatim"):
                continue
            if self._take(sentence):
                yield self._row(method, made[0], sentence, sentence, made[1])

    def _best_window(self, claim: Sentence, pool: Iterable[Sentence]) -> Sentence | None:
        words = self.words[claim.key]
        scored = [(overlap(words, self.words[s.key]), s) for s in pool if s.key not in self.used]
        # Near-duplicates (an abstract sentence repeated in the introduction) would make a "neutral" pair supported.
        scored = [p for p in scored if 0.0 < p[0] < NEAR_DUPLICATE]
        for _, target in sorted(scored, key=lambda p: (p[0], p[1].key), reverse=True):
            if _clean(claim.text) not in self._premise(target)["text"]:
                return target
        return None

    def generate(self, method: str) -> Iterable[dict[str, Any]]:
        if method == "sentence_verbatim":
            return self._same_sentence(method, lambda s: (s, {}))
        if method == "sentence_rewrite":
            return self._same_sentence(
                method, lambda s: (r, {"rewrite": "clause removed"}) if (r := rewrite(s)) else None
            )
        if method == "perturb_number":
            return self._same_sentence(method, lambda s: perturb_number(s, self.rng))
        if method == "perturb_negation":
            return self._same_sentence(method, perturb_negation)
        if method == "perturb_antonym":
            return self._same_sentence(method, perturb_antonym)
        if method == "perturb_swap":
            return self._same_sentence(method, perturb_swap)
        if method == "same_passage_far":
            return self._same_passage_far()
        if method == "same_paper_other_passage":
            return self._cross(method, same_paper=True)
        if method == "other_paper":
            return self._cross(method, same_paper=False)
        raise ValueError(method)

    def _same_passage_far(self) -> Iterable[dict[str, Any]]:
        for claim in self.candidates:
            sentences = self.by_chunk[claim.chunk.chunk_id]
            index = next(i for i, s in enumerate(sentences) if s.key == claim.key)
            far = [s for i, s in enumerate(sentences) if abs(i - index) >= 3 and len(s.text) >= 40]
            target = self._best_window(claim, far)
            if target is not None and self._take(claim, target):
                yield self._row("same_passage_far", claim.text, claim, target)

    def _cross(self, method: str, *, same_paper: bool) -> Iterable[dict[str, Any]]:
        by_paper: dict[str, list[Sentence]] = defaultdict(list)
        for chunk in self.chunks:
            by_paper[chunk.arxiv_id].extend(s for s in self.by_chunk[chunk.chunk_id] if len(s.text) >= 40)
        for claim in self.candidates:
            paper = claim.chunk.arxiv_id
            if same_paper:
                section = claim.chunk.section_path[:1]
                pool: Sequence[Sentence] = [
                    s
                    for s in by_paper[paper]
                    if s.chunk.chunk_id != claim.chunk.chunk_id and s.chunk.section_path[:1] != section
                ]
            else:
                pool = [s for other, items in sorted(by_paper.items()) if other != paper for s in items]
            target = self._best_window(claim, pool)
            if target is not None and self._take(claim, target):
                yield self._row(method, claim.text, claim, target)


def eligible_chunks(chunks: Iterable[DomainChunk], split: str = "val") -> list[DomainChunk]:
    return sorted(
        (
            c
            for c in chunks
            if c.split == split
            and c.section_kind in ("body", "abstract")
            and c.token_count >= 80
            and not any(re.search(r"reference|bibliograph|acknowledg", p, re.IGNORECASE) for p in c.section_path)
            and not reference_chunk(c.text)
        ),
        key=lambda c: c.chunk_id,
    )


def build_pairs(chunks: Iterable[DomainChunk], *, seed: int, split: str = "val") -> list[dict[str, Any]]:
    """About 150 pairs, balanced by expected label; shortfalls in one method are topped up within its label."""
    builder = Builder(eligible_chunks(chunks, split), seed)
    rows: list[dict[str, Any]] = []
    generators = {method: iter(builder.generate(method)) for method in QUOTAS}
    for method, (_, quota) in QUOTAS.items():
        rows.extend(row for _, row in zip(range(quota), generators[method], strict=False))
    for label in EXPECTED:
        want = sum(n for lab, n in QUOTAS.values() if lab == label)
        methods = [m for m, (lab, _) in QUOTAS.items() if lab == label]
        while sum(r["construction"]["expected"] == label for r in rows) < want and methods:
            for method in list(methods):
                row = next(generators[method], None)
                if row is None:
                    methods.remove(method)
                    continue
                rows.append(row)
                if sum(r["construction"]["expected"] == label for r in rows) >= want:
                    break
    builder.rng.shuffle(rows)
    return [{"pair_id": f"nli_{n:03d}", **row} for n, row in enumerate(rows, start=1)]


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "by_method": dict(sorted(Counter(r["construction"]["method"] for r in rows).items())),
        "by_expected_label": dict(sorted(Counter(r["construction"]["expected"] for r in rows).items())),
        "papers": len({r["premise"]["arxiv_id"] for r in rows} | {r["claim_source"]["arxiv_id"] for r in rows}),
    }
