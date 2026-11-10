"""Terminal review tool for the product owner (decision D2). Every keypress is saved immediately, so it can be closed
and resumed at any point. Labels are appended to `data/domain/v1/labels/<split>.human.jsonl`; nothing is overwritten.

On a random fifth of the questions the LLM pre-label is hidden, to measure how much the pre-labels anchor the reviewer.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mezvuat.evaluation.domain import DATASET_VERSION, DomainChunk, blind_query_ids, read_jsonl

TOOL_VERSION = "review-v1"
SECONDS_PER_ITEM = 30  # planning estimate; the range 25 to 40 s is used for the hour estimates
BLIND_FRACTION = 0.2
GRADE_NAMES = {0: "not relevant", 1: "partially relevant", 2: "answers the question"}


@dataclass(frozen=True)
class ReviewItem:
    query_id: str
    question: str
    chunk_id: str
    passage_id: str
    prelabel: int | None
    prelabel_reason: str | None
    blind: bool
    query_position: int
    query_count: int


def build_items(dataset_dir: Path, split: str, *, seed: int) -> list[ReviewItem]:
    queries = {q["query_id"]: q for q in read_jsonl(dataset_dir / "queries.jsonl") if q["split"] == split}
    prelabels = [p for p in read_jsonl(dataset_dir / "prelabels.jsonl") if p["split"] == split]
    blind = blind_query_ids(sorted(queries), fraction=BLIND_FRACTION, seed=seed)
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in prelabels:
        by_query.setdefault(row["query_id"], []).append(row)
    ordered = sorted(by_query)
    items: list[ReviewItem] = []
    for position, query_id in enumerate(ordered, start=1):
        rows = sorted(by_query[query_id], key=lambda r: int(r["passage_id"][1:]))
        for row in rows:
            items.append(
                ReviewItem(
                    query_id=query_id,
                    question=queries[query_id]["text"],
                    chunk_id=row["chunk_id"],
                    passage_id=row["passage_id"],
                    prelabel=row["grade"],
                    prelabel_reason=row["reason"],
                    blind=query_id in blind,
                    query_position=position,
                    query_count=len(ordered),
                )
            )
    return items


def effective_labels(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest record per (query, passage); a retraction removes the label."""
    current: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["query_id"], record["chunk_id"])
        if record.get("retracted"):
            current.pop(key, None)
        else:
            current[key] = record
    return current


class ReviewSession:
    def __init__(
        self,
        items: list[ReviewItem],
        labels_path: Path,
        *,
        reviewer: str,
        split: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.items = items
        self.path = labels_path
        self.reviewer = reviewer
        self.split = split
        self._clock = clock
        self.records = read_jsonl(labels_path)
        self.history: list[int] = []
        self.skipped: set[int] = set()
        self._started = clock()

    def labelled(self) -> dict[tuple[str, str], dict[str, Any]]:
        return effective_labels(self.records)

    def next_index(self) -> int | None:
        done = self.labelled()
        pending = [i for i, item in enumerate(self.items) if (item.query_id, item.chunk_id) not in done]
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

    def grade(self, index: int, grade: int, *, accepted_prelabel: bool = False) -> None:
        if grade not in (0, 1, 2):
            raise ValueError("grade must be 0, 1 or 2")
        item = self.items[index]
        now = self._clock()
        self._append(
            {
                "query_id": item.query_id,
                "chunk_id": item.chunk_id,
                "split": self.split,
                "grade": grade,
                "prelabel_shown": not item.blind,
                "prelabel_grade": item.prelabel,
                "accepted_prelabel": accepted_prelabel,
                "seconds": round(now - self._started, 1),
                "reviewer": self.reviewer,
                "labelled_at": datetime.now(UTC).isoformat(),
                "tool_version": TOOL_VERSION,
                "dataset_version": DATASET_VERSION,
            }
        )
        self._started = now
        self.history.append(index)

    def accept(self, index: int) -> bool:
        item = self.items[index]
        if item.blind or item.prelabel is None:
            return False
        self.grade(index, item.prelabel, accepted_prelabel=True)
        return True

    def undo(self) -> int | None:
        if not self.history:
            return None
        index = self.history.pop()
        item = self.items[index]
        self._append({"query_id": item.query_id, "chunk_id": item.chunk_id, "split": self.split, "retracted": True})
        self._started = self._clock()
        return index


def read_key() -> str:
    """One keypress without Enter where the terminal allows it."""
    if sys.platform == "win32":
        import msvcrt

        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            return ""
        return "\n" if key == "\r" else key
    if not sys.stdin.isatty():
        line = input()
        return line[:1] if line else "\n"
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
    return "\n" if key == "\r" else key


HELP = (
    "Keys:  0 / 1 / 2  grade the passage     Enter  accept the shown pre-label\n"
    "       u  undo the last grade    s  skip for now    q  quit (everything is already saved)    ?  this help\n"
    "Grades: 2 = answers the question   1 = partially relevant   0 = not relevant   (see docs/LABELLING_GUIDE.md)"
)


def render(item: ReviewItem, chunk: DomainChunk, *, done: int, total: int, passages_in_query: int) -> str:
    width = min(shutil.get_terminal_size((100, 40)).columns, 110) - 2
    where = f"{chunk.title} | {' > '.join(chunk.section_path) or 'no section'} | page {chunk.page_start or '?'}"
    lines = [
        f"[{done}/{total} labelled]  question {item.query_position}/{item.query_count}  "
        f"passage {item.passage_id} of {passages_in_query}",
        "",
        textwrap.fill(f"QUESTION: {item.question}", width),
        "",
        textwrap.fill(where, width),
        "-" * width,
        *(textwrap.fill(paragraph, width) for paragraph in chunk.text.split("\n\n")),
        "-" * width,
    ]
    if item.blind:
        lines.append("Pre-label hidden for this question. Please judge it yourself.")
    elif item.prelabel is None:
        lines.append("No pre-label (the model did not return one). Please judge it yourself.")
    else:
        lines.append(
            textwrap.fill(f"Pre-label: {item.prelabel} ({GRADE_NAMES[item.prelabel]}). {item.prelabel_reason}", width)
        )
    lines += ["", "0 / 1 / 2 = grade   Enter = accept pre-label   u = undo   s = skip   q = quit   ? = help"]
    return "\n".join(lines)


def run_review(split: str, *, reviewer: str, chunks: dict[str, DomainChunk], dataset_dir: Path, seed: int) -> None:
    items = build_items(dataset_dir, split, seed=seed)
    if not items:
        raise SystemExit(f"No pre-labelled items for split {split!r}.")
    for item in items:
        if item.chunk_id not in chunks:
            raise SystemExit("A pooled passage is missing from the local corpus. Rebuild it: mezvuat-dataset corpus")
    session = ReviewSession(items, dataset_dir / "labels" / f"{split}.human.jsonl", reviewer=reviewer, split=split)
    per_query = Counter(item.query_id for item in items)
    message = ""
    while True:
        index = session.next_index()
        if index is None:
            print(f"All {len(items)} items in {split} are labelled. Thank you.")
            return
        item = items[index]
        os.system("cls" if os.name == "nt" else "clear")  # noqa: S605  (fixed command, no input)
        print(
            render(
                item,
                chunks[item.chunk_id],
                done=len(session.labelled()),
                total=len(items),
                passages_in_query=per_query[item.query_id],
            )
        )
        if message:
            print(message)
            message = ""
        key = read_key().lower()
        if key in ("0", "1", "2"):
            session.grade(index, int(key))
        elif key == "\n":
            if not session.accept(index):
                message = "No pre-label to accept here: press 0, 1 or 2."
        elif key == "u":
            message = "Undone." if session.undo() is not None else "Nothing to undo in this session."
        elif key == "s":
            session.skipped.add(index)
        elif key == "q":
            print(f"Saved. {len(session.labelled())}/{len(items)} labelled. Run the same command to continue.")
            return
        elif key == "?":
            message = HELP


def status_report(dataset_dir: Path, *, seed: int) -> str:
    lines = [f"Dataset {DATASET_VERSION} in {dataset_dir}"]
    queries = read_jsonl(dataset_dir / "queries.jsonl")
    lines.append("Queries: " + ", ".join(f"{s}={n}" for s, n in sorted(Counter(q["split"] for q in queries).items())))
    for split in ("val", "test"):
        items = build_items(dataset_dir, split, seed=seed)
        if not items:
            continue
        labelled = effective_labels(read_jsonl(dataset_dir / "labels" / f"{split}.human.jsonl"))
        remaining = len(items) - len(labelled)
        blind = sum(1 for i in items if i.blind)
        lines.append(
            f"{split}: {len(items)} items over {len({i.query_id for i in items})} questions, {blind} with hidden "
            f"pre-labels; {len(labelled)} labelled; about {remaining * 25 / 3600:.1f} to "
            f"{remaining * 40 / 3600:.1f} hours left"
        )
        if labelled:
            shown = [r for r in labelled.values() if r["prelabel_shown"] and r["prelabel_grade"] is not None]
            hidden = [r for r in labelled.values() if not r["prelabel_shown"] and r["prelabel_grade"] is not None]
            for name, rows in (("pre-label shown", shown), ("pre-label hidden", hidden)):
                if rows:
                    agree = sum(r["grade"] == r["prelabel_grade"] for r in rows) / len(rows)
                    lines.append(f"  agreement with the LLM, {name}: {agree:.1%} of {len(rows)}")
    return "\n".join(lines)
