"""Research runs: start them in a bounded thread pool, fan their progress out to SSE subscribers, persist results.

Runs execute in the API process. The graph is synchronous (see `inference/types.py`), so each run holds one
worker thread; `RUN_CONCURRENCY` bounds how many run at once, and further runs queue. Progress events are kept
in memory per run so that a client can reconnect with `Last-Event-ID`; after a restart, the persisted run row
is the source of truth.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver

from mezvuat.core.ids import new_id
from mezvuat.inference import DataPolicy
from mezvuat.orchestration import Budget, ResearchState, RunEvent, RunFilters, WorkflowDeps, build_graph, run_to_end
from mezvuat.retrieval.types import RetrievalMode

log = logging.getLogger(__name__)
TERMINAL_EVENTS = frozenset({"run_completed", "run_failed"})


@dataclass(frozen=True)
class BusEvent:
    id: int
    event: str
    data: dict[str, Any]


@dataclass
class _Channel:
    events: list[BusEvent] = field(default_factory=list)
    closed: bool = False


class RunEventBus:
    """Per-run append-only event log with blocking reads. Keeps the most recent `capacity` runs."""

    def __init__(self, capacity: int = 200) -> None:
        self._channels: OrderedDict[str, _Channel] = OrderedDict()
        self._condition = threading.Condition()
        self._capacity = capacity

    def open(self, run_id: str) -> None:
        with self._condition:
            self._channels[run_id] = _Channel()
            while len(self._channels) > self._capacity:
                self._channels.popitem(last=False)

    def has(self, run_id: str) -> bool:
        with self._condition:
            return run_id in self._channels

    def publish(self, run_id: str, event: str, data: dict[str, Any]) -> None:
        with self._condition:
            channel = self._channels.get(run_id)
            if channel is None or channel.closed:
                return
            channel.events.append(BusEvent(len(channel.events) + 1, event, data))
            if event in TERMINAL_EVENTS:
                channel.closed = True
            self._condition.notify_all()

    def read(self, run_id: str, after: int, timeout: float) -> tuple[list[BusEvent], bool]:
        """Events with id > `after`, waiting up to `timeout` for some. Returns (events, closed)."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                channel = self._channels.get(run_id)
                if channel is None:
                    return [], True
                fresh = channel.events[after:]
                if fresh or channel.closed:
                    return fresh, channel.closed
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], False
                self._condition.wait(remaining)


class RunStore(Protocol):
    def create(
        self, *, run_id: str, workspace_id: str, question: str, data_policy: str, retrieval_mode: str
    ) -> None: ...

    def mark_running(self, run_id: str) -> None: ...

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        answer: dict[str, Any] | None,
        summary: dict[str, Any],
        error_code: str | None,
        error_message: str | None,
    ) -> None: ...

    def get(self, workspace_id: str, run_id: str) -> dict[str, Any] | None: ...


# Builds the workflow dependencies for one run (retrieval session, LLM client, recorder) and releases them after.
WorkflowFactory = Callable[[str, str, RetrievalMode], AbstractContextManager[WorkflowDeps]]


@dataclass(frozen=True)
class RunLimits:
    deadline_s: float
    max_llm_calls: int
    max_tokens: int


class RunService:
    def __init__(
        self,
        *,
        store: RunStore,
        workflow: WorkflowFactory,
        checkpointer: Callable[[], BaseCheckpointSaver[Any]],
        limits: RunLimits,
        concurrency: int,
        bus: RunEventBus | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.bus = bus or RunEventBus()
        self._workflow = workflow
        self._checkpointer = checkpointer
        self._limits = limits
        self._executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="research-run")
        self._clock = clock

    def start(
        self, *, workspace_id: str, data_policy: DataPolicy, question: str, mode: RetrievalMode, filters: RunFilters
    ) -> str:
        run_id = new_id("run")
        self.store.create(
            run_id=run_id, workspace_id=workspace_id, question=question, data_policy=data_policy, retrieval_mode=mode
        )
        self.bus.open(run_id)
        self.bus.publish(run_id, "run_queued", {"run_id": run_id})
        self._executor.submit(self._execute, run_id, workspace_id, data_policy, question, mode, filters)
        return run_id

    def _execute(
        self,
        run_id: str,
        workspace_id: str,
        data_policy: DataPolicy,
        question: str,
        mode: RetrievalMode,
        filters: RunFilters,
    ) -> None:
        try:
            self.store.mark_running(run_id)
            self.bus.publish(run_id, "run_started", {"run_id": run_id})
            initial = ResearchState(
                run_id=run_id,
                workspace_id=workspace_id,
                question=question,
                data_policy=data_policy,
                filters=filters,
                # The deadline starts when the run starts, not when it was queued.
                budget=Budget(
                    llm_calls_left=self._limits.max_llm_calls,
                    tokens_left=self._limits.max_tokens,
                    deadline=self._clock() + self._limits.deadline_s,
                ),
            )

            def forward(event: RunEvent) -> None:
                self.bus.publish(run_id, event.event, event.data)

            with self._workflow(run_id, workspace_id, mode) as deps:
                graph = build_graph(deps, self._checkpointer())
                final = run_to_end(graph, initial, forward)
        except Exception as exc:  # A bug or an unexpected outage: record it, never leave the run "running".
            log.exception("research run crashed", extra={"run_id": run_id})
            message = f"internal error ({type(exc).__name__})"
            self._finish_safely(run_id, "failed", None, {}, "internal_error", message)
            self.bus.publish(run_id, "run_failed", {"run_id": run_id, "code": "internal_error", "message": message})
            return
        summary = summarize(final)
        if final.status == "completed" and final.answer is not None:
            answer = final.answer.model_dump(mode="json")
            self._finish_safely(run_id, "completed", answer, summary, None, None)
            self.bus.publish(run_id, "run_completed", {"run_id": run_id, "answer": answer, "summary": summary})
        else:
            error = final.error
            code = error.code if error else "run_failed"
            message = error.message if error else "the run did not produce an answer"
            self._finish_safely(run_id, "failed", None, summary, code, message)
            self.bus.publish(run_id, "run_failed", {"run_id": run_id, "code": code, "message": message})

    def _finish_safely(self, run_id: str, status: str, *args: Any) -> None:
        answer, summary, code, message = args
        try:
            self.store.finish(
                run_id, status=status, answer=answer, summary=summary, error_code=code, error_message=message
            )
        except Exception:
            log.exception("could not persist the run result", extra={"run_id": run_id})

    def events(self, run_id: str, after: int, timeout: float) -> tuple[list[BusEvent], bool]:
        return self.bus.read(run_id, after, timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def summarize(state: ResearchState) -> dict[str, Any]:
    providers = sorted({c.provider for c in state.llm_calls})
    return {
        "iterations": state.iteration,
        "resynthesized": state.resynth_count > 0,
        "evidence_items": len(state.evidence),
        "retrieval_calls": len(state.retrieval_log),
        "llm_calls": len(state.llm_calls),
        "llm_calls_left": state.budget.llm_calls_left,
        "backend": "direct",
        "providers": providers,
        "prompt_tokens": sum(c.prompt_tokens or 0 for c in state.llm_calls),
        "completion_tokens": sum(c.completion_tokens or 0 for c in state.llm_calls),
    }


def sse_format(event: BusEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"id: {event.id}\nevent: {event.event}\ndata: {payload}\n\n"
