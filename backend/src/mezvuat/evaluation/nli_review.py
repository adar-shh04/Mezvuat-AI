"""The owner's labelling of NLI calibration pairs (`review nli`). Resumable and append-only.

Labels go to `data/domain/nli/v1/review.human.jsonl`; the relevance labels in `data/domain/v1/labels/` and the
temporal review file are never touched. **No model suggestion is shown, for any pair** (D2 amendment, 2026-11-14:
the owner's labels anchored heavily on shown suggestions). The screen has the claim, the evidence text and where
it comes from. It does not show how the pair was built, the label that construction expects, or any NLI score.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mezvuat.evaluation.domain import read_jsonl, sha256_text
from mezvuat.evaluation.review import read_key

TOOL_VERSION = "nli-review-v1"
PAIRS_FILE = "pairs.jsonl"
REVIEW_FILE = "review.human.jsonl"
LABELS = {"y": "supported", "c": "contradicted", "n": "neutral"}


@dataclass(frozen=True)
class NliPair:
    data: dict[str, Any]

    @property
    def pair_id(self) -> str:
        return str(self.data["pair_id"])

    @property
    def fingerprint(self) -> str:
        """Ties a label to the exact text shown, so an edited pair is never matched to an old label."""
        return sha256_text(self.data["claim"] + "\n" + self.data["premise"]["text"])[:16]


def load_pairs(nli_dir: Path) -> list[NliPair]:
    return [NliPair(row) for row in sorted(read_jsonl(nli_dir / PAIRS_FILE), key=lambda r: r["pair_id"])]


def current_labels(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("retracted"):
            current.pop(record["pair_id"], None)
        else:
            current[record["pair_id"]] = record
    return current


class NliReviewSession:
    def __init__(
        self, pairs: list[NliPair], path: Path, *, reviewer: str, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.pairs = pairs
        self.path = path
        self.reviewer = reviewer
        self._clock = clock
        self.records = read_jsonl(path)
        self.history: list[int] = []
        self.skipped: set[int] = set()
        self._started = clock()

    def labelled(self) -> dict[str, dict[str, Any]]:
        fingerprints = {p.pair_id: p.fingerprint for p in self.pairs}
        return {
            pair_id: record
            for pair_id, record in current_labels(self.records).items()
            if fingerprints.get(pair_id) == record.get("fingerprint")
        }

    def next_index(self) -> int | None:
        done = self.labelled()
        pending = [i for i, pair in enumerate(self.pairs) if pair.pair_id not in done]
        fresh = [i for i in pending if i not in self.skipped]
        if fresh:
            return fresh[0]
        self.skipped.clear()
        return pending[0] if pending else None

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)

    def label(self, index: int, label: str) -> None:
        """`label` is supported, contradicted, neutral, or `unusable` for a pair that cannot be judged."""
        if label not in (*LABELS.values(), "unusable"):
            raise ValueError(label)
        pair = self.pairs[index]
        now = self._clock()
        self._append(
            {
                "pair_id": pair.pair_id,
                "label": label,
                "fingerprint": pair.fingerprint,
                "suggestion_shown": False,
                "seconds": round(now - self._started, 1),
                "reviewer": self.reviewer,
                "labelled_at": datetime.now(UTC).isoformat(),
                "tool_version": TOOL_VERSION,
            }
        )
        self._started = now
        self.history.append(index)

    def undo(self) -> int | None:
        if not self.history:
            return None
        index = self.history.pop()
        self._append({"pair_id": self.pairs[index].pair_id, "retracted": True})
        self._started = self._clock()
        return index


def render(pair: NliPair, *, done: int, total: int) -> str:
    """What the reviewer sees. Deliberately nothing from `construction` and no model output."""
    width = min(shutil.get_terminal_size((100, 40)).columns, 110) - 2
    premise = pair.data["premise"]
    where = " > ".join(premise.get("section_path") or []) or "no section"
    page = premise.get("page_start") or "?"
    source = f"Evidence from: {premise['title']} (arXiv {premise['arxiv_id']}) | {where} | page {page}"
    return "\n".join(
        [
            f"[{done}/{total} labelled]  {pair.pair_id}",
            "",
            textwrap.fill(f"CLAIM: {pair.data['claim']}", width),
            "",
            textwrap.fill(source, width),
            "-" * width,
            textwrap.fill(premise["text"], width),
            "-" * width,
            "Does the evidence text above support the claim?",
            "y = supported   c = contradicted   n = neither (not enough information)   x = cannot judge (broken text)",
            "u = undo   s = skip   q = quit   ? = help",
        ]
    )


HELP = (
    "Judge the claim against the evidence text only, not against what you know about the paper.\n"
    "y: the evidence states the claim or it follows directly.  c: the evidence says something that cannot be true\n"
    "at the same time as the claim.  n: neither.  x: the text is too garbled to judge.  Everything is saved at once."
)


def run_nli_review(
    nli_dir: Path,
    *,
    reviewer: str,
    key: Callable[[], str] = read_key,
    clear: bool = True,
) -> None:
    pairs = load_pairs(nli_dir)
    if not pairs:
        raise SystemExit("No NLI calibration pairs yet. Build them first: mezvuat-dataset nli-pairs")
    session = NliReviewSession(pairs, nli_dir / REVIEW_FILE, reviewer=reviewer)
    message = ""
    while True:
        index = session.next_index()
        if index is None:
            print(f"All {len(pairs)} NLI pairs are labelled. Thank you.")
            return
        if clear:
            os.system("cls" if os.name == "nt" else "clear")  # noqa: S605  (fixed command, no input)
        print(render(pairs[index], done=len(session.labelled()), total=len(pairs)))
        if message:
            print(message)
            message = ""
        pressed = key().lower()
        if pressed in LABELS:
            session.label(index, LABELS[pressed])
        elif pressed == "x":
            session.label(index, "unusable")
        elif pressed == "u":
            message = "Undone." if session.undo() is not None else "Nothing to undo in this session."
        elif pressed == "s":
            session.skipped.add(index)
        elif pressed == "q":
            print(f"Saved. {len(session.labelled())}/{len(pairs)} labelled. Run the same command to continue.")
            return
        elif pressed == "?":
            message = HELP
        else:
            message = "Press y, c, n or x (or ? for help)."


def nli_status(nli_dir: Path) -> str:
    pairs = load_pairs(nli_dir)
    if not pairs:
        return "NLI calibration pairs: none built yet."
    session = NliReviewSession(pairs, nli_dir / REVIEW_FILE, reviewer="status")
    labelled = session.labelled()
    counts = Counter(r["label"] for r in labelled.values())
    remaining = len(pairs) - len(labelled)
    line = (
        f"NLI calibration pairs: {len(pairs)}, {len(labelled)} labelled"
        f" ({', '.join(f'{k}={n}' for k, n in sorted(counts.items())) or 'none'}); "
        f"about {remaining * 20 / 3600:.1f} to {remaining * 35 / 3600:.1f} hours left (estimate: 20 to 35 s a pair)"
    )
    return line
