"""The owner's review of the drafted temporal questions (decision D11). Resumable, append-only.

Decisions are appended to `data/domain/temporal/v1/review.human.jsonl`; the relevance labels in
`data/domain/v1/labels/` are never touched. For a random fifth of the items the draft answer is hidden, and the
reviewer writes the answer and its behaviour (answerable or "can't be determined") themselves. Agreement on the
behaviour, with and without the draft shown, measures anchoring (D2).
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

from mezvuat.evaluation.domain import blind_query_ids, read_jsonl
from mezvuat.evaluation.review import read_key

TOOL_VERSION = "temporal-review-v1"
BLIND_FRACTION = 0.2
REVIEW_FILE = "review.human.jsonl"


@dataclass(frozen=True)
class TemporalItem:
    data: dict[str, Any]
    blind: bool

    @property
    def item_id(self) -> str:
        return str(self.data["item_id"])


def load_items(temporal_dir: Path, *, seed: int) -> list[TemporalItem]:
    rows = {row["item_id"]: row for row in read_jsonl(temporal_dir / "drafts.jsonl")}
    blind = blind_query_ids(sorted(rows), fraction=BLIND_FRACTION, seed=seed)
    return [TemporalItem(rows[i], i in blind) for i in sorted(rows)]


def current_decisions(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("retracted"):
            current.pop(record["item_id"], None)
        else:
            current[record["item_id"]] = record
    return current


class TemporalReviewSession:
    def __init__(
        self, items: list[TemporalItem], path: Path, *, reviewer: str, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.items = items
        self.path = path
        self.reviewer = reviewer
        self._clock = clock
        self.records = read_jsonl(path)
        self.history: list[int] = []
        self.skipped: set[int] = set()
        self._started = clock()

    def decided(self) -> dict[str, dict[str, Any]]:
        return current_decisions(self.records)

    def next_index(self) -> int | None:
        done = self.decided()
        pending = [i for i, item in enumerate(self.items) if item.item_id not in done]
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

    def decide(self, index: int, decision: str, *, answer: str | None = None, behavior: str | None = None) -> None:
        if decision not in ("accept", "edit", "reject", "write"):
            raise ValueError(decision)
        item = self.items[index]
        if decision == "accept" and item.blind:
            raise ValueError("the draft is hidden for this item; write the answer instead")
        if decision in ("edit", "write") and (not answer or behavior not in ("answer", "abstain")):
            raise ValueError("an edited or written item needs an answer and a behaviour")
        draft_behavior = item.data["expected_behavior"]
        final_behavior = draft_behavior if decision == "accept" else behavior
        now = self._clock()
        self._append(
            {
                "item_id": item.item_id,
                "decision": decision,
                "answer": item.data["draft_answer"] if decision == "accept" else answer,
                "expected_behavior": final_behavior,
                "draft_shown": not item.blind,
                "draft_behavior": draft_behavior,
                "agrees_with_draft_behavior": None if decision == "reject" else final_behavior == draft_behavior,
                "seconds": round(now - self._started, 1),
                "reviewer": self.reviewer,
                "reviewed_at": datetime.now(UTC).isoformat(),
                "tool_version": TOOL_VERSION,
            }
        )
        self._started = now
        self.history.append(index)

    def undo(self) -> int | None:
        if not self.history:
            return None
        index = self.history.pop()
        self._append({"item_id": self.items[index].item_id, "retracted": True})
        self._started = self._clock()
        return index


def render(item: TemporalItem, *, done: int, total: int) -> str:
    width = min(shutil.get_terminal_size((100, 40)).columns, 110) - 2
    data = item.data
    versions = ", ".join(f"{v['label']} submitted {v['submitted']}" for v in data["versions"])
    lines = [
        f"[{done}/{total} reviewed]  {data['item_id']}  kind: {data['kind']}",
        "",
        textwrap.fill(f"QUESTION: {data['question']}", width),
        "",
        textwrap.fill(f"Paper: {data['title']} (arXiv {data['arxiv_id']}). Versions: {versions}", width),
    ]
    if data.get("as_of"):
        lines.append(f"As-of date in the question: {data['as_of']}")
    lines.append("-" * width)
    for n, ev in enumerate(data["evidence"], start=1):
        lines.append(f"Evidence {n}: {ev['arxiv_id']}{ev['version']} (submitted {ev['submitted']}), page {ev['page']}")
        lines.append(textwrap.fill(f'"{ev["quote"]}"', width, initial_indent="  ", subsequent_indent="  "))
    lines.append("-" * width)
    if item.blind:
        lines += [
            "Draft answer hidden for this item. Write the answer yourself.",
            "",
            "w = write answer   r = reject the question   u = undo   s = skip   q = quit   ? = help",
        ]
    else:
        lines += [
            textwrap.fill(f"DRAFT ANSWER: {data['draft_answer']}", width),
            f"Expected behaviour: {data['expected_behavior']}  "
            f"(versions: {', '.join(data['expected_versions']) or '-'})",
            "",
            "Enter = accept   e = edit   r = reject   u = undo   s = skip   q = quit   ? = help",
        ]
    return "\n".join(lines)


HELP = (
    "Accept when the question is clear, the answer is right and fully supported by the evidence shown.\n"
    "Edit to correct the answer (and say whether it is answerable or can't be determined). Reject a question that\n"
    "is unclear, trivial, or not about a real change between versions. Everything is saved at once."
)


def ask_answer(prompt: Callable[[str], str]) -> tuple[str, str] | None:
    answer = prompt("Answer (empty to cancel): ").strip()
    if not answer:
        return None
    behavior = prompt("Answerable from the evidence (a) or can't be determined (n)? ").strip().lower()
    if behavior not in ("a", "n"):
        return None
    return answer, "answer" if behavior == "a" else "abstain"


def run_temporal_review(
    temporal_dir: Path,
    *,
    reviewer: str,
    seed: int,
    key: Callable[[], str] = read_key,
    prompt: Callable[[str], str] = input,
    clear: bool = True,
) -> None:
    items = load_items(temporal_dir, seed=seed)
    if not items:
        raise SystemExit("No temporal drafts yet. Draft them first: mezvuat-dataset temporal-draft")
    session = TemporalReviewSession(items, temporal_dir / REVIEW_FILE, reviewer=reviewer)
    message = ""
    while True:
        index = session.next_index()
        if index is None:
            print(f"All {len(items)} temporal items are reviewed. Thank you.")
            return
        item = items[index]
        if clear:
            os.system("cls" if os.name == "nt" else "clear")  # noqa: S605  (fixed command, no input)
        print(render(item, done=len(session.decided()), total=len(items)))
        if message:
            print(message)
            message = ""
        pressed = key().lower()
        if pressed == "\n" and not item.blind:
            session.decide(index, "accept")
        elif pressed in ("e", "w") and (pressed == "w") == item.blind:
            written = ask_answer(prompt)
            if written is None:
                message = "Cancelled."
            else:
                session.decide(index, "write" if item.blind else "edit", answer=written[0], behavior=written[1])
        elif pressed == "r":
            session.decide(index, "reject")
        elif pressed == "u":
            message = "Undone." if session.undo() is not None else "Nothing to undo in this session."
        elif pressed == "s":
            session.skipped.add(index)
        elif pressed == "q":
            print(f"Saved. {len(session.decided())}/{len(items)} reviewed. Run the same command to continue.")
            return
        elif pressed == "?":
            message = HELP
        else:
            message = "Write the answer with w (draft hidden here)." if item.blind else "Press Enter, e or r."


def temporal_status(temporal_dir: Path, *, seed: int) -> str:
    items = load_items(temporal_dir, seed=seed)
    if not items:
        return "Temporal set: no drafts yet."
    decided = current_decisions(read_jsonl(temporal_dir / REVIEW_FILE))
    kinds = Counter(i.data["kind"] for i in items)
    lines = [
        f"Temporal set: {len(items)} drafts ({', '.join(f'{k}={n}' for k, n in sorted(kinds.items()))}), "
        f"{sum(i.blind for i in items)} with hidden drafts; {len(decided)} reviewed",
    ]
    if decided:
        lines.append(
            "  decisions: "
            + ", ".join(f"{k}={n}" for k, n in sorted(Counter(r["decision"] for r in decided.values()).items()))
        )
        for shown in (True, False):
            rows = [
                r for r in decided.values() if r["draft_shown"] is shown and r["agrees_with_draft_behavior"] is not None
            ]
            if rows:
                agree = sum(r["agrees_with_draft_behavior"] for r in rows) / len(rows)
                lines.append(
                    f"  behaviour agreement, draft {'shown' if shown else 'hidden'}: {agree:.1%} of {len(rows)}"
                )
    return "\n".join(lines)
