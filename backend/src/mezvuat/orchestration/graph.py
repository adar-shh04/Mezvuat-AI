"""The research graph (ADR-0008) and the run driver that streams its progress.

```
START -> understand_query -> plan_research -> retrieve -> assess_evidence
assess_evidence --(insufficient & iteration < MAX_ITER & budget ok)--> reformulate -> retrieve
assess_evidence --(sufficient | limits hit)--> synthesize -> verify
verify --(unsupported claims & resynth_count < 1 & budget ok)--> synthesize
verify --(ok | resynth limit)--> respond -> END
any node --(fatal)--> fail -> END
```
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from mezvuat.orchestration.deps import WorkflowDeps
from mezvuat.orchestration.nodes import ResearchNodes
from mezvuat.orchestration.state import RECURSION_LIMIT, STATE_MODELS, ResearchState

NODES = (
    "understand_query",
    "plan_research",
    "retrieve",
    "assess_evidence",
    "reformulate",
    "synthesize",
    "verify",
    "respond",
    "fail",
)


def checkpoint_serializer() -> JsonPlusSerializer:
    """Only our state classes may be revived from a checkpoint (LangGraph's msgpack allowlist)."""
    return JsonPlusSerializer(allowed_msgpack_modules=[(m.__module__, m.__name__) for m in STATE_MODELS])


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=checkpoint_serializer())


def build_graph(deps: WorkflowDeps, checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledStateGraph[Any]:
    nodes = ResearchNodes(deps)
    graph = StateGraph(ResearchState)
    graph.add_node("understand_query", nodes.understand_query)
    graph.add_node("plan_research", nodes.plan_research)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("assess_evidence", nodes.assess_evidence)
    graph.add_node("reformulate", nodes.reformulate)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("verify", nodes.verify)
    graph.add_node("respond", nodes.respond)
    graph.add_node("fail", nodes.fail)

    graph.add_edge(START, "understand_query")
    graph.add_conditional_edges("understand_query", nodes.or_fail("plan_research"), ["plan_research", "fail"])
    graph.add_conditional_edges("plan_research", nodes.or_fail("retrieve"), ["retrieve", "fail"])
    graph.add_conditional_edges("retrieve", nodes.or_fail("assess_evidence"), ["assess_evidence", "fail"])
    graph.add_conditional_edges("assess_evidence", nodes.route_after_assess, ["reformulate", "synthesize", "fail"])
    graph.add_conditional_edges("reformulate", nodes.route_after_reformulate, ["retrieve", "synthesize", "fail"])
    graph.add_conditional_edges("synthesize", nodes.or_fail("verify"), ["verify", "fail"])
    graph.add_conditional_edges("verify", nodes.route_after_verify, ["synthesize", "respond", "fail"])
    graph.add_edge("respond", END)
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=checkpointer or memory_checkpointer())


def run_config(run_id: str) -> RunnableConfig:
    # Set explicitly: LangGraph's default (10,007) is far too permissive to act as a guard. The longest legal path
    # is 15 steps (3 + 2 loops x 3 + synthesize, verify, synthesize, verify, respond).
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


@dataclass(frozen=True)
class RunEvent:
    """One progress event: `node` (a node finished) or a custom event emitted by a node (answer deltas)."""

    event: str
    data: dict[str, Any]


def stream_run(graph: CompiledStateGraph[Any], initial: ResearchState) -> Iterator[RunEvent]:
    """Run to completion, yielding progress. The final state is read back from the checkpointer."""
    config = run_config(initial.run_id)
    for mode, chunk in graph.stream(initial, config, stream_mode=["updates", "custom"]):
        if mode == "updates":
            for node in chunk:
                yield RunEvent("node", {"node": node})
        elif mode == "custom" and isinstance(chunk, dict):
            data = dict(chunk)
            yield RunEvent(str(data.pop("event", "custom")), data)


def final_state(graph: CompiledStateGraph[Any], run_id: str) -> ResearchState:
    values = graph.get_state(run_config(run_id)).values
    return ResearchState.model_validate(values)


def run_to_end(
    graph: CompiledStateGraph[Any], initial: ResearchState, on_event: Callable[[RunEvent], None] | None = None
) -> ResearchState:
    for event in stream_run(graph, initial):
        if on_event is not None:
            on_event(event)
    return final_state(graph, initial.run_id)
