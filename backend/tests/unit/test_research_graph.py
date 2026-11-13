"""Research graph tests with a scripted LLM (ADR-0008 verification plan, M4 acceptance D7 and D8)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mezvuat.citations import PageSpan
from mezvuat.inference import TASK_TYPES, InferenceUnavailableError, LLMRequest, StreamIncompleteError
from mezvuat.orchestration import (
    MAX_ITER,
    RECURSION_LIMIT,
    Budget,
    ResearchState,
    RunEvent,
    WorkflowDeps,
    build_graph,
    run_config,
    run_to_end,
)
from mezvuat.orchestration.graph import NODES
from tests.fakes import KeywordSearch, ListRecorder, ScriptedLLM, StaticPages, chunk, id_for

BM25 = (
    "BM25 is a bag-of-words ranking function. It scores documents by term frequency and inverse document "
    "frequency, with length normalisation controlled by the parameter b."
)
DENSE = "Dense retrieval encodes queries and passages into vectors and ranks passages by inner product similarity."
RRF = "Reciprocal rank fusion combines ranked lists by summing the reciprocal of each rank plus a constant k."
OTHER_WS = "BM25 secret from another workspace: term frequency documents ranking."

CHUNKS = [
    chunk("c_bm25", BM25, doc="doc_bm25", version="ver_bm25", char_start=100, page_start=1, page_end=2),
    chunk("c_dense", DENSE, doc="doc_dense", version="ver_dense", char_start=0),
    chunk("c_rrf", RRF, doc="doc_rrf", version="ver_rrf", char_start=0, page_start=3, page_end=3),
    chunk("c_other", OTHER_WS, workspace_id="ws_2", doc="doc_x", version="ver_x"),
]
# Page 1 ends inside the BM25 chunk (absolute offset 150), so a quote near its end is on page 2.
PAGES = StaticPages({"ver_bm25": [PageSpan(page=1, start=0, end=150), PageSpan(page=2, start=150, end=400)]})


def analysis(*, single_hop: bool) -> dict[str, Any]:
    return {
        "objective": "understand ranking",
        "subject": "retrieval",
        "constraints": [],
        "expected_answer_type": "explanation",
        "temporal_scope": {"as_of": None, "wants_latest": False, "compare_versions": False},
        "needs_multi_doc": not single_hop,
        "single_hop": single_hop,
    }


def plan(*questions: str) -> dict[str, Any]:
    return {
        "subquestions": [{"id": f"S{i}", "question": q} for i, q in enumerate(questions, start=1)],
        "success_criteria": "definitions",
    }


def coverage(*covered: bool) -> dict[str, Any]:
    return {
        "subquestions": [
            {"subquestion_id": f"S{i}", "covered": c, "missing": "" if c else "details"}
            for i, c in enumerate(covered, start=1)
        ],
        "rationale": "scripted",
    }


def claim(claim_id: str, text: str, evidence_id: str, quote: str, kind: str = "fact") -> dict[str, Any]:
    return {
        "id": claim_id,
        "text": text,
        "evidence_ids": [evidence_id],
        "quotes": [{"evidence_id": evidence_id, "text": quote}],
        "kind": kind,
    }


def good_draft(request: LLMRequest) -> dict[str, Any]:
    bm25 = id_for(request, "BM25 is a bag-of-words")
    return {
        "claims": [
            claim(
                "C1",
                "BM25 scores documents using term frequency and inverse document frequency.",
                bm25,
                "It scores documents by term frequency and inverse document frequency",
            )
        ],
        "conflicts": [],
        "gaps": [],
    }


def answer_citing(*phrases: str) -> Any:
    def build(request: LLMRequest) -> str:
        markers = " ".join(f"[{_marker_for(request, p)}]" for p in phrases)
        return f"BM25 ranks documents by term statistics {markers}."

    return build


def _marker_for(request: LLMRequest, phrase: str) -> str:
    # The respond prompt lists verified claims with their markers, not evidence blocks.
    for line in request.messages[-1].content.splitlines():
        if phrase in line and "[E" in line:
            return line[line.index("[E") + 1 : line.index("]", line.index("[E"))]
    raise AssertionError(f"claim {phrase!r} not in the respond prompt")


def state(budget_calls: int = 12, deadline: float = 10_000.0, question: str = "How does BM25 rank documents?"):
    return ResearchState(
        run_id="run_01TEST",
        workspace_id="ws_1",
        question=question,
        data_policy="any",
        budget=Budget(llm_calls_left=budget_calls, tokens_left=1_000_000, deadline=deadline),
    )


class Clock:
    def __init__(self, start: float = 0.0, step: float = 0.0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def run(llm: ScriptedLLM, initial: ResearchState | None = None, **deps: Any) -> tuple[ResearchState, list[RunEvent]]:
    events: list[RunEvent] = []
    search = deps.pop("search", KeywordSearch(CHUNKS))
    workflow = WorkflowDeps(llm=llm, search=search, pages=PAGES, clock=deps.pop("clock", Clock()), **deps)
    graph = build_graph(workflow)
    final = run_to_end(graph, initial or state(), events.append)
    return final, events


def nodes_visited(events: list[RunEvent]) -> list[str]:
    return [e.data["node"] for e in events if e.event == "node"]


# ----- edges ---------------------------------------------------------------------------------------------------------


def test_single_hop_happy_path_visits_every_stage_once() -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    recorder = ListRecorder()
    final, events = run(llm, recorder=recorder)
    assert nodes_visited(events) == [
        "understand_query",
        "plan_research",
        "retrieve",
        "assess_evidence",
        "synthesize",
        "verify",
        "respond",
    ]
    assert final.status == "completed"
    assert final.answer is not None and final.answer.rendering == "model"
    # Shortcut: a single-hop plan needs no planning call.
    assert llm.calls("research_planning") == 0
    assert [r.task_type for r in final.llm_calls] == [
        "query_classification",
        "evidence_assessment",
        "evidence_synthesis",
        "final_response",
    ]
    assert recorder.rows == final.llm_calls
    assert {r.task_type for r in final.llm_calls} <= TASK_TYPES
    assert len({r.client_request_id for r in final.llm_calls}) == len(final.llm_calls)
    assert all(r.prompt_version.count("@") == 1 for r in final.llm_calls)
    assert any(e.event == "answer_delta" for e in events)
    # Workspace isolation holds inside the graph too.
    assert all(item.chunk_id != "c_other" for item in final.evidence.values())


def test_multi_hop_plan_reformulates_once_then_answers() -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=False)],
            "research_planning": [plan("What is BM25?", "What is reciprocal rank fusion?")],
            "evidence_assessment": [coverage(True, False), coverage(True, True)],
            "query_rewriting": [{"rewrites": [{"subquestion_id": "S2", "new_queries": ["fusion constant k"]}]}],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    final, events = run(llm)
    assert nodes_visited(events) == [
        "understand_query",
        "plan_research",
        "retrieve",
        "assess_evidence",
        "reformulate",
        "retrieve",
        "assess_evidence",
        "synthesize",
        "verify",
        "respond",
    ]
    assert final.iteration == 1
    assert [row.query for row in final.retrieval_log][-1] == "fusion constant k"
    assert final.status == "completed"


def test_max_iter_stops_the_research_loop() -> None:
    always_missing = coverage(False)
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        },
        default={
            "evidence_assessment": always_missing,
            "query_rewriting": lambda r: {
                "rewrites": [{"subquestion_id": "S1", "new_queries": [f"bm25 variant {len(r.messages[-1].content)}"]}]
            },
        },
    )
    final, events = run(llm)
    assert nodes_visited(events).count("reformulate") == MAX_ITER
    assert final.iteration == MAX_ITER
    assert llm.calls("evidence_assessment") == MAX_ITER + 1
    assert final.status == "completed"
    assert any("Evidence may be incomplete" in line for line in final.answer.limitations)  # type: ignore[union-attr]


def test_reformulation_without_new_queries_goes_to_synthesis() -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(False)],
            # Only an already-tried query: nothing new to search for.
            "query_rewriting": [
                {"rewrites": [{"subquestion_id": "S1", "new_queries": ["How does BM25 rank documents?"]}]}
            ],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    final, events = run(llm)
    visited = nodes_visited(events)
    assert visited[visited.index("reformulate") + 1] == "synthesize"
    assert final.status == "completed"


def test_budget_exhaustion_reaches_respond_with_limitations() -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=False)],
            "research_planning": [plan("What is BM25?")],
            "evidence_assessment": [coverage(False)],
            "evidence_synthesis": [good_draft],
        }
    )
    # 4 calls: understand, plan, assess, synthesize. The loop needs 4 more, the response 1 more.
    final, events = run(llm, state(budget_calls=4))
    visited = nodes_visited(events)
    assert "reformulate" not in visited
    assert visited[-1] == "respond"
    assert llm.calls("final_response") == 0
    assert final.budget.llm_calls_left == 0
    assert final.status == "completed"
    answer = final.answer
    assert answer is not None
    assert answer.rendering == "deterministic"
    assert "budget_exhausted_before_response" in answer.flags
    assert answer.citations, "the deterministic rendering still cites the verified claims"


def test_deadline_exhaustion_skips_llm_steps_and_abstains() -> None:
    llm = ScriptedLLM({"query_classification": [analysis(single_hop=True)]})
    # Each clock read advances 30 s against a 60 s deadline: after the first call nothing else may run.
    final, events = run(llm, state(deadline=60.0), clock=Clock(step=30.0))
    assert nodes_visited(events)[-1] == "respond"
    assert llm.calls("evidence_synthesis") == 0
    assert llm.calls("final_response") == 0
    answer = final.answer
    assert answer is not None and "abstained" in answer.flags
    assert answer.citations == []
    assert any("budget was exhausted" in line for line in answer.limitations)


def test_retrieval_failure_goes_to_fail() -> None:
    llm = ScriptedLLM({"query_classification": [analysis(single_hop=True)]})
    final, events = run(llm, search=KeywordSearch(CHUNKS, fail=True))
    assert nodes_visited(events) == ["understand_query", "plan_research", "retrieve", "fail"]
    assert final.status == "failed"
    assert final.error is not None and final.error.code == "retrieval_unavailable"
    assert final.answer is None
    assert llm.calls("evidence_synthesis") == 0


def test_inference_unavailable_goes_to_fail_and_still_records_the_call() -> None:
    llm = ScriptedLLM({"query_classification": [InferenceUnavailableError("down")]})
    recorder = ListRecorder()
    final, events = run(llm, recorder=recorder)
    assert nodes_visited(events) == ["understand_query", "fail"]
    assert final.status == "failed"
    assert final.error is not None and final.error.code == "inference_unavailable"
    assert len(recorder.rows) == 1 and recorder.rows[0].status == "error"


def test_structured_output_failure_degrades_instead_of_failing() -> None:
    from mezvuat.inference import StructuredOutputInvalidError

    llm = ScriptedLLM(
        {
            "query_classification": [StructuredOutputInvalidError("bad json")],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    final, _ = run(llm)
    assert final.status == "completed"
    assert final.analysis is None
    assert any("Query analysis was skipped" in line for line in final.answer.limitations)  # type: ignore[union-attr]


def test_failed_verification_triggers_one_resynthesis() -> None:
    def fabricated(request: LLMRequest) -> dict[str, Any]:
        return {"claims": [claim("C1", "BM25 was invented in 2030.", "E99", "BM25 was invented in the year 2030")],
                "conflicts": [], "gaps": []}  # fmt: skip

    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [fabricated, good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    final, events = run(llm)
    visited = nodes_visited(events)
    assert visited.count("synthesize") == 2 and visited.count("verify") == 2
    assert final.resynth_count == 1
    second = [r for r in llm.requests if r.task_type == "evidence_synthesis"][1]
    assert "failed verification" in second.messages[-1].content
    assert "E99 is not in the evidence registry" in second.messages[-1].content
    assert final.answer is not None and final.answer.rendering == "model"


def test_resynthesis_limit_drops_unverifiable_claims() -> None:
    def paraphrased(request: LLMRequest) -> dict[str, Any]:
        bm25 = id_for(request, "BM25 is a bag-of-words")
        return {"claims": [claim("C1", "BM25 uses word counts.", bm25, "BM25 counts words and weights rare ones")],
                "conflicts": [], "gaps": []}  # fmt: skip

    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [paraphrased, paraphrased],
        }
    )
    final, events = run(llm)
    assert nodes_visited(events).count("synthesize") == 2
    answer = final.answer
    assert answer is not None
    assert answer.markdown.startswith("I could not find evidence")
    assert "abstained" in answer.flags
    assert any("could not verify this claim" in line and "word counts" in line for line in answer.limitations)
    assert llm.calls("final_response") == 0


def test_unknown_marker_in_the_answer_is_replaced_by_the_verified_claims() -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": ["BM25 is great [E1] and was proven optimal [E7]."],
        }
    )
    final, events = run(llm)
    answer = final.answer
    assert answer is not None
    assert answer.rendering == "deterministic"
    assert "unknown_citation_markers" in answer.flags
    assert "[E7]" not in answer.markdown
    assert "proven optimal" not in answer.markdown
    assert any(e.event == "answer_replaced" for e in events)


def test_interrupted_stream_is_discarded_and_retried_once() -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": [StreamIncompleteError("no [DONE]"), answer_citing("BM25 scores documents")],
        }
    )
    final, events = run(llm)
    kinds = [e.event for e in events]
    assert "answer_reset" in kinds
    assert llm.calls("final_response") == 2
    answer = final.answer
    assert answer is not None and answer.rendering == "model"
    assert "partial" not in answer.markdown
    assert "stream_interrupted" in answer.flags


# ----- citations (D8, partial) ---------------------------------------------------------------------------------------


def test_every_citation_marker_resolves_to_a_verified_span_on_the_right_page() -> None:
    def two_claims(request: LLMRequest) -> dict[str, Any]:
        bm25 = id_for(request, "BM25 is a bag-of-words")
        first = "BM25 is a bag-of-words ranking function"
        second = "length normalisation controlled by the parameter b"
        return {
            "claims": [
                claim("C1", "BM25 is a bag-of-words ranking function.", bm25, first),
                claim("C2", "Its length normalisation is set by b.", bm25, second),
            ],
            "conflicts": [],
            "gaps": [],
        }

    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [two_claims],
            "final_response": [answer_citing("bag-of-words ranking", "length normalisation")],
        }
    )
    final, _ = run(llm)
    answer = final.answer
    assert answer is not None and answer.rendering == "model"
    import re

    markers = set(re.findall(r"\[(E\d+)\]", answer.markdown))
    assert markers
    for marker in markers:
        assert marker in final.evidence, "every marker is a registry id"
        cited = [c for c in answer.citations if c.evidence_id == marker]
        assert cited, f"{marker} resolves to at least one citation"
        for citation in cited:
            item = final.evidence[marker]
            assert citation.chunk_id == item.chunk_id
            local = (citation.char_start - item.char_start, citation.char_end - item.char_start)
            assert item.text[local[0] : local[1]] == citation.quote
            assert citation.quote in item.text
    pages = {c.quote[:20]: c.page for c in answer.citations}
    assert pages["BM25 is a bag-of-wor"] == 1
    assert pages["length normalisation"] == 2


def test_claims_citing_ids_the_synthesis_prompt_did_not_show_are_rejected() -> None:
    def cites_unseen(request: LLMRequest) -> dict[str, Any]:
        shown = set(__import__("tests.fakes", fromlist=["shown_evidence"]).shown_evidence(request))
        unseen = next(f"E{n}" for n in range(1, 50) if f"E{n}" not in shown)
        return {"claims": [claim("C1", "Something.", unseen, "BM25 is a bag-of-words ranking function")],
                "conflicts": [], "gaps": []}  # fmt: skip

    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [cites_unseen, cites_unseen],
        }
    )
    final, _ = run(llm)
    assert final.verification and all(v.status == "unsupported" for v in final.verification)
    assert final.answer is not None and final.answer.citations == []


# ----- structure and runtime -----------------------------------------------------------------------------------------


def test_recursion_limit_is_set_explicitly() -> None:
    config = run_config("run_x")
    assert config["recursion_limit"] == RECURSION_LIMIT == 30


def test_graph_cycles_all_pass_through_a_bounded_counter() -> None:
    graph = build_graph(WorkflowDeps(llm=ScriptedLLM({}), search=KeywordSearch([]))).get_graph()
    assert set(graph.nodes) == {"__start__", "__end__", *NODES}
    successors: dict[str, set[str]] = {}
    for edge in graph.edges:
        successors.setdefault(edge.source, set()).add(edge.target)

    cycles: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str]) -> None:
        for nxt in successors.get(node, set()):
            if nxt in path:
                cycle = path[path.index(nxt) :]
                start = cycle.index(min(cycle))
                cycles.add(tuple(cycle[start:] + cycle[:start]))
            else:
                walk(nxt, [*path, nxt])

    walk("__start__", ["__start__"])
    # reformulate increments `iteration` (bounded by MAX_ITER); synthesize increments `resynth_count` when a draft
    # already exists (bounded by MAX_RESYNTH). No other cycle exists.
    assert cycles == {("assess_evidence", "reformulate", "retrieve"), ("synthesize", "verify")}


def test_checkpointed_state_round_trips_without_unregistered_types(caplog: pytest.LogCaptureFixture) -> None:
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    with caplog.at_level(logging.WARNING):
        final, _ = run(llm)
    assert "unregistered type" not in caplog.text
    assert isinstance(final, ResearchState)
    assert final.answer is not None and final.answer.citations
