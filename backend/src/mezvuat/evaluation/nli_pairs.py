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

Perturbations must change what the claim says, so they skip (v2, after a review of v1):
- numbers inside `[...]` citations, years, section/table/figure numbers, versions, licences, dataset names and
  venues ("Apache 2.0", "SQuAD 2.0", "CIKM '26"), ordinals, dash ranges, ratios, and digits that are part of a
  larger grouped number (a grouped number is changed whole or not at all); measured numbers (%, x, points,
  tokens, ...) are preferred, and an unmeasured number is used only in a comparative sentence;
- deictic and positional words (above, below, earlier, later, former, latter), fixed terms ("large language
  models", "high-level"), idioms ("no longer", "how many", "to better ...") and discourse markers ("More broadly,");
- swaps unless two distinct systems stand on either side of a comparative relation in the same clause (not
  metrics, not names that contain each other, not a name that is the object of a preposition);
- negations next to a word that already negates ("insufficient", "irrelevant", "less"), in sentences that are
  already negative, or where "have"/"do" is the main verb.
Sentences and chunks with glued text (a run of 28 or more letters: lost spaces from PDF extraction) are excluded,
and so are quoted prompt templates ("You are ..."). `rejection_audit` counts each rule over the whole pool.
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

GENERATOR_VERSION = "nli-pairs-v2"
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
        and not GLUED.search(text)
        and not re.match(r"(?:You|Your)\b", text)  # prompt templates quoted in papers
    )


# Text with its spaces lost in PDF extraction ("retrievalandgeneration:theretrieverruns..."); see the M5 report.
GLUED = re.compile(r"[A-Za-z]{28,}")
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


def _reject(stats: Counter[str] | None, reason: str) -> None:
    if stats is not None:
        stats[reason] += 1


# A whole number: digits, optional thousands groups, optional decimals. Never part of a longer token or list.
_NUMBER = re.compile(r"(?<![\w.,'\u2019\-/])(?<!\d:)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)(?![\w\u2019/]|[.,:]\d)")
# "2-4 documents" (hyphen or en dash): changing one end of a range can produce "4-4".
# Ratios such as "94/120" and "1:1:1" are excluded by the pattern itself.
_RANGE_BEFORE = re.compile(r"\d\s*[\u2013\u2014-]\s*$")
_RANGE_AFTER = re.compile(r"^\s*[\u2013\u2014-]\s*\d")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_NOT_A_QUANTITY = re.compile(
    r"(?:Table|Figure|Fig\.|Section|Sec\.|Eq\.|Equation|Appendix|Algorithm|Step|Chapter|Line|Lines|Stage|Phase)\s*$",
    re.IGNORECASE,
)
# Words that make the next number a name, version or venue: "Apache 2.0", "version 3", "SQuAD 2.0", "CIKM '26".
_IDENTIFIER_WORD_LIST = (
    "version versions release licence license apache python cuda torch pytorch v ver rev revision gpt "
    "llama qwen mistral gemma phi bert t5 squad hotpotqa trec clef beir mteb ms marco cikm sigir acl "
    "emnlp naacl eacl coling neurips iclr icml kdd wsdm ecir www aaai ijcai lrec interspeech iso rfc "
    "ieee"
)
_IDENTIFIER_WORDS = frozenset(_IDENTIFIER_WORD_LIST.split())
_MEASURE_AFTER = re.compile(
    r"^\s*(?:%|\u00d7|x\b|points?\b|pp\b|percentage points\b|tokens?\b|ms\b|s\b|seconds?\b|minutes?\b|hours?\b|"
    r"GB\b|MB\b|"
    r"k\b|K\b|M\b|times\b|fold\b|queries\b|documents\b|passages\b|papers\b|examples\b|samples\b|items\b|"
    r"questions\b|words\b|characters\b|dimensions\b|layers\b|parameters\b|nDCG|MRR|F1|EM\b|recall|accuracy|"
    r"BLEU|ROUGE|on\b|higher|lower|better|worse|more|fewer|less)",
)
_MEASURE_BEFORE = re.compile(
    r"(?:\bby|\bof|\bto|\bat|\breaches|\breached|\bachieves|\bachieved|\bscores?|\bimproves?|\bgains?|"
    r"\bdrops?|\breduces?|\bincreases?|\bfrom|\bonly|\babout|\baround|\bapproximately|\bnearly|\bover|"
    r"\bunder|\bwithin|\bof\s+the|[+±~≈:])\s*$",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:than|outperform\w*|surpass\w*|exceed\w*|beats?|higher|lower|better|worse|improv\w+|"
    r"increase\w*|decrease\w*|reduc\w+)\b",
    re.IGNORECASE,
)


def _previous_word(text: str) -> str:
    match = re.search(r"([A-Za-z][A-Za-z0-9-]*)\W*$", text)
    return match.group(1) if match else ""


def _new_number(whole: str, decimals: str, rng: random.Random) -> str | None:
    grouped = "," in whole
    value = int(whole.replace(",", ""))
    if decimals:
        places = len(decimals) - 1
        number = float(f"{value}{decimals}")
        factor = rng.choice((0.55, 0.7, 1.4, 1.6))
        if number <= 100 < number * factor:
            factor = 0.7  # keep scores on a 0-100 scale plausible
        new = f"{number * factor:.{places}f}"
        return None if new == f"{value}{decimals}" else new
    if value < 2:
        return None
    changed = rng.choice((value * 2, max(value // 2, 1), value + 3 + value // 4))
    return f"{changed:,}" if grouped else str(changed)


def perturb_number(
    sentence: str, rng: random.Random, stats: Counter[str] | None = None
) -> tuple[str, dict[str, str]] | None:
    """Change one measured number. Citations, years, versions, identifiers and list items are left alone."""
    brackets = [(m.start(), m.end()) for m in _BRACKETS.finditer(sentence)]
    candidates: list[tuple[int, re.Match[str]]] = []
    for match in _NUMBER.finditer(sentence):
        whole, decimals, percent = match.group(1), match.group(2) or "", match.group(3)
        before, after = sentence[: match.start()], sentence[match.end() :]
        if any(a <= match.start() < b for a, b in brackets):
            _reject(stats, "number:inside_citation_brackets")
            continue
        if _NOT_A_QUANTITY.search(before):
            _reject(stats, "number:section_table_or_figure_number")
            continue
        if not decimals and not percent and "," not in whole and 1900 <= int(whole) <= 2099:
            _reject(stats, "number:year")
            continue
        word = _previous_word(before)
        if word.lower() in _IDENTIFIER_WORDS or re.fullmatch(r"[vV]\d*", word) or before.rstrip().endswith(("v", "V")):
            _reject(stats, "number:version_licence_dataset_or_venue")
            continue
        if word and not percent and re.search(r"[A-Z]", word[1:]) and not _MEASURE_AFTER.match(after):
            # "SQuAD 2.0", "MS MARCO 1": a number right after a name is part of the name.
            _reject(stats, "number:after_a_name")
            continue
        if re.match(r"\s*(?:st|nd|rd|th)\b", after):
            _reject(stats, "number:ordinal")
            continue
        if _RANGE_BEFORE.search(before) or _RANGE_AFTER.match(after):
            _reject(stats, "number:end_of_a_dash_range")
            continue
        measured = bool(percent) or bool(_MEASURE_AFTER.match(after)) or bool(_MEASURE_BEFORE.search(before))
        if not measured and not _COMPARISON.search(sentence):
            _reject(stats, "number:not_a_measurement_or_comparison")
            continue
        candidates.append((0 if measured else 1, match))
    for _, match in sorted(candidates, key=lambda c: (c[0], c[1].start())):
        whole, decimals, percent = match.group(1), match.group(2) or "", match.group(3)
        new = _new_number(whole, decimals, rng)
        if new is None or (percent and float(new.replace(",", "")) > 100):
            _reject(stats, "number:no_valid_replacement")
            continue
        old = whole + decimals
        changed = sentence[: match.start()] + new + sentence[match.start() + len(old) :]
        return changed, {"from": old + percent, "to": new + percent}
    return None


_REMOVE_NOT = re.compile(r"\b(is|are|was|were|does|do|did|can|could|will|would|should|may|has|have)\s+not\b")
_CANNOT = re.compile(r"\bcannot\b")
_AUX = re.compile(r"\b(is|are|was|were|can|could|will|would|should|does|do|has|have|may)\s+(?!not\b)([A-Za-z-]+)")
_NEGATIVE = re.compile(
    r"\b(?:no|not|never|none|neither|nor|without|cannot|lack|lacks|lacking|absent|fails?|failed)\b|n't\b", re.IGNORECASE
)
_NEGATIVE_NEXT = frozenset({"less", "least", "hardly", "rarely", "seldom", "few", "little", "barely"})
# Words that already negate: "insufficient", "unclear", "irrelevant", "non-trivial", "useless".
_PARTICIPLE = re.compile(r"^(?:\w+(?:ed|en)|been|shown|made|done|built|set|put|run|found|held|kept|led|seen|known)$")
_NEGATIVE_PREFIX = re.compile(r"^(?:un|in|im|ir|il|non|dis|mis)[a-z-]{3,}$|^[a-z]{3,}less$", re.IGNORECASE)


def perturb_negation(sentence: str, stats: Counter[str] | None = None) -> tuple[str, dict[str, str]] | None:
    match = _REMOVE_NOT.search(sentence)
    if match:
        return sentence[: match.start()] + match.group(1) + sentence[match.end() :], {"from": "not", "to": ""}
    match = _CANNOT.search(sentence)
    if match:
        return sentence[: match.start()] + "can" + sentence[match.end() :], {"from": "cannot", "to": "can"}
    if _NEGATIVE.search(sentence):
        _reject(stats, "negation:sentence_already_negative")
        return None  # adding a second negation reads as nonsense, not as a contradiction
    for match in _AUX.finditer(sentence):
        if match.start() == 0 or match.group(2).endswith("ly"):
            continue
        following = sentence[match.end(1) :].split()[:2]
        words = [w.strip(".,;:()").lower() for w in following]
        if any(_NEGATIVE_PREFIX.match(w) or w in _NEGATIVE_NEXT for w in words):
            _reject(stats, "negation:negative_word_at_insertion_point")
            return None
        auxiliary, verb = match.group(1), match.group(2).lower()
        if auxiliary in ("do", "does") or (auxiliary in ("has", "have") and not _PARTICIPLE.match(verb)):
            # "have judged-positive identifiers" is "have" as a main verb; "not" there is ungrammatical.
            _reject(stats, "negation:have_or_do_as_main_verb")
            return None
        position = match.end(1)
        return sentence[:position] + " not" + sentence[position:], {
            "from": match.group(1),
            "to": match.group(1) + " not",
        }
    _reject(stats, "negation:no_auxiliary")
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
)
# Deictic and positional words point into the document ("described above"); swapping them changes no claim.
DEICTIC = frozenset({"above", "below", "earlier", "later", "former", "latter"})
_FIXED_TERM = re.compile(
    r"(?:large\s+language|small\s+language|(?:high|low)(?:er)?[\s-]+level|more\s+or\s+less|"
    r"(?:more|less|most)\s+\w+ly\s*,|at\s+least|at\s+most|the\s+most\s+part)",
    re.IGNORECASE,
)
_ANTONYMS = dict(_ANTONYM_PAIRS) | {b: a for a, b in _ANTONYM_PAIRS if b not in {"increases"}}
_ANTONYM_RE = re.compile(
    r"\b(" + "|".join(sorted(set(_ANTONYMS) | DEICTIC, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def perturb_antonym(sentence: str, stats: Counter[str] | None = None) -> tuple[str, dict[str, str]] | None:
    for match in _ANTONYM_RE.finditer(sentence):
        word = match.group(1)
        if word.lower() in DEICTIC:
            _reject(stats, "antonym:deictic_or_positional_word")
            continue
        if _FIXED_TERM.match(sentence[match.start() :]) or re.search(
            r"\b(?:no|any|how|to)\s+$", sentence[: match.start()], re.IGNORECASE
        ):
            # "large language models" is a name; "no longer", "how many" and "to better exploit" are idioms;
            # "More broadly," is a discourse marker.
            _reject(stats, "antonym:fixed_term_idiom_or_discourse_marker")
            continue
        replacement = _ANTONYMS[word.lower()]
        if word[0].isupper():
            replacement = _capitalise(replacement)
        return sentence[: match.start()] + replacement + sentence[match.end() :], {"from": word, "to": replacement}
    _reject(stats, "antonym:no_directional_word")
    return None


_SYSTEM_NAME = re.compile(r"\b[A-Za-z]*[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9-]*\b")
# A comparison between two named systems: "X outperforms Y", "X is better than Y", "X achieves higher F1 than Y".
_COMPARATIVE_RELATION = re.compile(
    r"\b(?:(?:consistently|significantly|clearly|substantially)\s+)?"
    r"(?:outperform(?:s|ed)?|underperform(?:s|ed)?|surpass(?:es|ed)?|beat(?:s)?|exceed(?:s|ed)?|"
    r"(?:better|worse|higher|lower|faster|slower|stronger|weaker|larger|smaller|more\s+\w+|less\s+\w+)"
    r"(?:\s+[\w@-]+){0,2}\s+than|(?:is|are|was|were)\s+(?:superior|inferior)\s+to)\b",
    re.IGNORECASE,
)


# Metrics look like system names ("F1", "MRR") but are what is compared, not who.
_METRIC_NAME_LIST = "F1 EM MRR MAP NDCG AUC ACC BLEU ROUGE ROUGE-L METEOR BERTSCORE MSE RMSE MAE PPL WER CER P@1 R@1"
_METRIC_NAMES = frozenset(_METRIC_NAME_LIST.split())


def _same_clause(gap: str) -> bool:
    """Between a name and the comparison: at most four words and no clause punctuation."""
    return not re.search(r"[;:.,()]", gap) and len(gap.split()) <= 4


def perturb_swap(sentence: str, stats: Counter[str] | None = None) -> tuple[str, dict[str, str]] | None:
    """Swap the two systems on either side of a comparison. Names containing each other are not distinct."""
    relation = _COMPARATIVE_RELATION.search(sentence)
    if relation is None:
        _reject(stats, "swap:no_comparative_relation")
        return None
    names = [
        m for m in _SYSTEM_NAME.finditer(sentence) if len(m.group(0)) >= 2 and m.group(0).upper() not in _METRIC_NAMES
    ]
    counts = Counter(m.group(0) for m in names)
    before = [m for m in names if m.end() <= relation.start() and _same_clause(sentence[m.end() : relation.start()])]
    after = [m for m in names if m.start() >= relation.end() and _same_clause(sentence[relation.end() : m.start()])]
    if not before or not after:
        _reject(stats, "swap:no_name_on_both_sides")
        return None
    first, second = before[-1], after[0]
    lead = sentence[: first.start()]
    if re.search(r"(?:-\s*|\b(?:on|in|of|for|with|from|by|to|at|over|under|than)\s+)$", lead, re.IGNORECASE):
        # "fine-tuned on NASA- EO-Bench beats ...": the name is an object or a fragment, not the compared system.
        _reject(stats, "swap:first_name_is_not_the_subject")
        return None
    a, b = first.group(0), second.group(0)
    if counts[a] != 1 or counts[b] != 1:
        _reject(stats, "swap:name_repeated")
        return None
    if a.lower() in b.lower() or b.lower() in a.lower():
        _reject(stats, "swap:names_contain_each_other")
        return None
    swapped = sentence[: first.start()] + b + sentence[first.end() : second.start()] + a + sentence[second.end() :]
    return swapped, {"from": f"{a} ... {b}", "to": f"{b} ... {a}"}


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


def eligible_chunks(
    chunks: Iterable[DomainChunk], split: str = "val", *, exclude_glued: bool = True
) -> list[DomainChunk]:
    return sorted(
        (
            c
            for c in chunks
            if c.split == split
            and c.section_kind in ("body", "abstract")
            and c.token_count >= 80
            and not any(re.search(r"reference|bibliograph|acknowledg", p, re.IGNORECASE) for p in c.section_path)
            and not reference_chunk(c.text)
            and not (exclude_glued and GLUED.search(c.text))
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


def rejection_audit(chunks: Iterable[DomainChunk], *, seed: int, split: str = "val") -> dict[str, Any]:
    """How often each skip rule fires over the whole candidate pool, not just the sentences the sample reached.

    Counts are per candidate: per number occurrence for the number rules, per sentence for the others.
    """
    items = list(chunks)
    eligible = eligible_chunks(items, split)
    before_glued = eligible_chunks(items, split, exclude_glued=False)
    glued = [c for c in before_glued if GLUED.search(c.text)]
    sentences = [s.text for c in eligible for s in sentences_of(c) if good_sentence(s.text)]
    rng = random.Random(f"nli-audit:{seed}")  # noqa: S311  (reproducible)
    stats: Counter[str] = Counter()
    produced: Counter[str] = Counter()
    for text in sentences:
        made = {
            "perturb_number": perturb_number(text, rng, stats),
            "perturb_negation": perturb_negation(text, stats),
            "perturb_antonym": perturb_antonym(text, stats),
            "perturb_swap": perturb_swap(text, stats),
        }
        produced.update(method for method, result in made.items() if result is not None)
    return {
        "candidate_sentences": len(sentences),
        "glued_text_chunks_excluded": len(glued),
        "glued_text_papers": sorted({c.title for c in glued}),
        "rejections": dict(sorted(stats.items())),
        "perturbations_possible": dict(sorted(produced.items())),
    }


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "by_method": dict(sorted(Counter(r["construction"]["method"] for r in rows).items())),
        "by_expected_label": dict(sorted(Counter(r["construction"]["expected"] for r in rows).items())),
        "papers": len({r["premise"]["arxiv_id"] for r in rows} | {r["claim_source"]["arxiv_id"] for r in rows}),
    }
