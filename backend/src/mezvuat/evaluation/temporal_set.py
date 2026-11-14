"""The M5 temporal question set (decision D11): drafted by an LLM from the public corpus, reviewed by the owner.

Everything except the wording of the questions and answers is deterministic:
- Only papers with two or more arXiv versions are used. Versions and their dates come from the manifest (arXiv
  metadata, day precision); nothing is invented.
- Candidate changes come from a sentence-level diff of the first and latest version, parsed with the production
  PDF parser.
- The as-of dates are computed from the version dates, and so is the expected behaviour (answer from version X,
  or "can't be determined").
- Every quote the model returns is checked verbatim against the cited version's text and resolved to offsets and a
  page. An item with any failing quote is dropped.

Kinds:
- `as_of`: a date on which only the first version existed. The gold answer comes from v1.
- `latest`: the gold answer comes from the latest version.
- `version_change`: what changed between the first and the latest version, quoting both.
- `undeterminable`: a date before the first version, or the day two versions were both submitted. Gold: abstain.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from mezvuat.citations import find_quote
from mezvuat.ingestion.parsers.pdf import PdfParser

PROMPT_VERSION = "temporal-draft-v2"
SET_VERSION = "temporal-v1"
MAX_CANDIDATES = 6
MAX_QUOTE_CHARS = 450
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class VersionText:
    label: str
    submitted: date
    text: str
    pages: list[tuple[int, int, int]]  # (page, start, end)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    old: str
    new: str
    numbers_changed: bool


def load_version(pdf_path: Path, label: str, submitted: str, expected_sha256: str | None) -> VersionText:
    raw = pdf_path.read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{pdf_path.name} does not match the manifest hash")
    parsed = PdfParser().parse(raw)
    return VersionText(
        label=label,
        submitted=date.fromisoformat(submitted),
        text=parsed.text,
        pages=[(p.page_number, p.char_start, p.char_end) for p in parsed.pages],
    )


def sentences(text: str) -> list[str]:
    out = []
    for paragraph in text.split("\n\n"):
        for sentence in _SENTENCE.split(" ".join(paragraph.split())):
            if 40 <= len(sentence) <= 400 and sum(c.isalpha() for c in sentence) > len(sentence) / 2:
                out.append(sentence)
    return out


def _key(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def change_candidates(old_text: str, new_text: str, limit: int = MAX_CANDIDATES) -> list[Candidate]:
    """Sentence pairs that were reworded substantively between versions; changed numbers rank first."""
    old, new = sentences(old_text), sentences(new_text)
    matcher = difflib.SequenceMatcher(None, [_key(s) for s in old], [_key(s) for s in new], autojunk=False)
    pairs: list[tuple[float, str, str, bool]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        block = new[j1:j2]
        for offset, a in enumerate(old[i1:i2]):
            # Compare against a window around the proportional position: a reflowed section can make a block
            # hundreds of sentences long, and an all-pairs search would be quadratic in it.
            centre = offset * len(block) // max(i2 - i1, 1)
            window = block[max(centre - 8, 0) : centre + 9]
            # seq2 is the one SequenceMatcher caches, so it holds the fixed sentence.
            matcher_a = difflib.SequenceMatcher(None, "", _key(a), autojunk=False)
            ratio, best = 0.0, ""
            for b in window:
                matcher_a.set_seq1(_key(b))
                if matcher_a.real_quick_ratio() < 0.45 or matcher_a.quick_ratio() < 0.45:
                    continue
                score = matcher_a.ratio()
                if score > ratio:
                    ratio, best = score, b
            if not 0.45 <= ratio < 0.97:
                continue
            numbers = set(_DIGITS.findall(a)) != set(_DIGITS.findall(best))
            score = (3.0 if numbers else 0.0) + min(len(a), 300) / 300 + (1 - ratio)
            pairs.append((score, a, best, numbers))
    pairs.sort(key=lambda p: -p[0])
    seen: set[str] = set()
    chosen: list[Candidate] = []
    for _, a, b, numbers in pairs:
        if _key(a) in seen:
            continue
        seen.add(_key(a))
        chosen.append(Candidate(f"C{len(chosen) + 1}", a, b, numbers))
        if len(chosen) == limit:
            break
    return chosen


@dataclass(frozen=True)
class DatePlan:
    """Dates for the as-of and undeterminable questions, derived from the version dates only."""

    as_of: date | None  # a day on which v1 applied and v2 did not exist yet
    before_first: date  # a day before any version existed
    same_day: date | None  # v1 and v2 submitted on the same day: an as-of query that day cannot be resolved


def plan_dates(first: date, second: date) -> DatePlan:
    gap = (second - first).days
    if gap <= 0:
        return DatePlan(as_of=None, before_first=first - timedelta(days=30), same_day=first)
    as_of = first + timedelta(days=gap // 2) if gap >= 2 else first
    return DatePlan(as_of=as_of, before_first=first - timedelta(days=30), same_day=None)


SYSTEM = (
    "You write evaluation questions about how a research paper changed between its arXiv versions. "
    "Use only the sentences you are given. Copy quotes exactly, character for character. Never invent numbers."
)


def draft_prompt(title: str, versions: list[VersionText], candidates: list[Candidate], plan: DatePlan) -> str:
    first, latest = versions[0], versions[-1]
    lines = [
        f'Paper: "{title}"',
        "Versions: " + ", ".join(f"{v.label} submitted {v.submitted.isoformat()}" for v in versions),
        "",
        f"Sentences that changed between {first.label} (OLD) and {latest.label} (NEW):",
    ]
    for c in candidates:
        lines += [f"{c.candidate_id} OLD: {c.old}", f"{c.candidate_id} NEW: {c.new}", ""]
    as_of = plan.as_of.isoformat() if plan.as_of else (plan.same_day.isoformat() if plan.same_day else "")
    lines += [
        "Pick the one candidate with the clearest factual change (prefer changed numbers, settings or claims).",
        "Write three standalone questions that name the paper by its title:",
        f"- as_of_question: what the paper stated about the topic as of {as_of}. Answer from the OLD sentence.",
        f"- latest_question: what the latest version ({latest.label}) states about the topic. Answer from NEW.",
        f"- change_question: what changed about the topic between {first.label} and {latest.label}.",
        f"- before_question: the same topic, asked as of {plan.before_first.isoformat()}.",
        "Answers are one or two sentences. old_quote is copied exactly from the chosen OLD sentence and new_quote "
        "exactly from the chosen NEW sentence; each quote is 20 to 300 characters, contains the changed part, "
        "and keeps every symbol exactly as written.",
    ]
    return "\n".join(lines)


def draft_schema() -> dict[str, Any]:
    fields = [
        "candidate_id",
        "topic",
        "as_of_question",
        "as_of_answer",
        "latest_question",
        "latest_answer",
        "change_question",
        "change_answer",
        "before_question",
        "old_quote",
        "new_quote",
    ]
    return {
        "type": "object",
        "properties": {f: {"type": "string"} for f in fields},
        "required": fields,
        "additionalProperties": False,
    }


def standalone_question(question: str, title: str) -> str:
    """The model often writes "the paper" without naming it; retrieval needs the title, so it is prefixed."""
    text = question.strip()
    if title.lower()[:25] in text.lower():
        return text
    return f'In the paper "{title}": {text}'


def span(version: VersionText, quote: str, arxiv_id: str) -> dict[str, Any] | None:
    found = find_quote(version.text, quote)
    # Up to a whole candidate sentence: this is reviewer evidence, not a synthesis quote (ADR-0009's 300 cap).
    if found is None or not 20 <= len(quote.strip()) <= MAX_QUOTE_CHARS:
        return None
    start, end = found
    page = next((p for p, a, b in version.pages if a <= start < b), None)
    return {
        "arxiv_id": arxiv_id,
        "version": version.label,
        "submitted": version.submitted.isoformat(),
        "quote": version.text[start:end],
        "char_start": start,
        "char_end": end,
        "page": page,
    }


def build_items(
    *,
    arxiv_id: str,
    title: str,
    split: str,
    versions: list[VersionText],
    plan: DatePlan,
    draft: dict[str, Any],
    include_undeterminable: bool,
    drafted_by: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Items whose quotes all verify, and the reasons for anything dropped."""
    first, latest = versions[0], versions[-1]
    old_span = span(first, draft["old_quote"], arxiv_id)
    new_span = span(latest, draft["new_quote"], arxiv_id)
    problems = []
    if old_span is None:
        problems.append("old_quote is not verbatim in the first version")
    if new_span is None:
        problems.append("new_quote is not verbatim in the latest version")
    if old_span and new_span and old_span["quote"] == new_span["quote"]:
        problems.append("the quotes are identical, so they do not show a change")
    if problems:
        return [], problems
    assert old_span is not None and new_span is not None  # noqa: S101  (narrowing after the check above)
    meta = {"versions": [{"label": v.label, "submitted": v.submitted.isoformat()} for v in versions]}
    base = {"arxiv_id": arxiv_id, "title": title, "split": split, "topic": draft["topic"], **meta}

    def item(kind: str, question: str, answer: str, **extra: Any) -> dict[str, Any]:
        return {
            "item_id": f"{arxiv_id}:{kind}",
            "set_version": SET_VERSION,
            "kind": kind,
            "question": standalone_question(question, title),
            "drafted_question": question.strip(),
            "draft_answer": answer.strip(),
            **base,
            **extra,
            "status": "draft",
            "drafted_by": drafted_by,
        }

    items = []
    if plan.as_of is not None:
        items.append(
            item(
                "as_of",
                draft["as_of_question"],
                draft["as_of_answer"],
                as_of=plan.as_of.isoformat(),
                expected_behavior="answer",
                expected_versions=[first.label],
                evidence=[old_span],
            )
        )
    items.append(
        item(
            "latest",
            draft["latest_question"],
            draft["latest_answer"],
            as_of=None,
            expected_behavior="answer",
            expected_versions=[latest.label],
            evidence=[new_span],
        )
    )
    items.append(
        item(
            "version_change",
            draft["change_question"],
            draft["change_answer"],
            as_of=None,
            expected_behavior="answer",
            expected_versions=[first.label, latest.label],
            evidence=[old_span, new_span],
        )
    )
    if plan.same_day is not None:
        same = [v.label for v in versions if v.submitted == plan.same_day]
        items.append(
            item(
                "undeterminable",
                draft["as_of_question"],
                f"Cannot be determined: versions {', '.join(same)} were both submitted on "
                f"{plan.same_day.isoformat()}, so which one applied on that day cannot be decided.",
                as_of=plan.same_day.isoformat(),
                expected_behavior="abstain",
                expected_versions=[],
                reason="same_day_versions",
                evidence=[old_span, new_span],
            )
        )
    elif include_undeterminable:
        items.append(
            item(
                "undeterminable",
                draft["before_question"],
                f"Cannot be determined: the earliest version ({first.label}) was submitted on "
                f"{first.submitted.isoformat()}, after {plan.before_first.isoformat()}, so no version of the paper "
                "existed on that date.",
                as_of=plan.before_first.isoformat(),
                expected_behavior="abstain",
                expected_versions=[],
                reason="before_first_version",
                evidence=[old_span],
            )
        )
    return items, []


def eligible_papers(manifest: dict[str, Any], splits: dict[str, str], allowed_splits: set[str]) -> list[dict[str, Any]]:
    """Allowlisted papers with two or more downloaded versions, in the requested family splits."""
    from mezvuat.ingestion.corpus.manifest import ALLOWED_LICENSES

    return [
        p
        for p in manifest["papers"]
        if p["license"] in ALLOWED_LICENSES
        and splits.get(p["arxiv_id"]) in allowed_splits
        and len(p["versions"]) >= 2
        and all(v.get("raw_sha256") for v in p["versions"])
    ]
