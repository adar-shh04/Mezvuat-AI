"""A full research run through a mocked ModelMesh (M8 acceptance C1, C3, C6 on the Mezvuat side).

The mocked ModelMesh answers from the same scripts as the graph tests, through the contract's wire format, so the run
exercises the real `ModelMeshClient`, the real graph and the real fallback path. Nothing leaves the process.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import respx

from mezvuat.api.runs import summarize
from mezvuat.inference import ChatMessage, Correlation, LLMRequest
from mezvuat.orchestration import WorkflowDeps, build_graph, run_to_end
from mezvuat.orchestration.models import Draft, JudgeOutput, QueryAnalysis, Reformulation, ResearchPlan, Sufficiency
from tests.fakes import KeywordSearch, ListRecorder, ScriptedLLM
from tests.modelmesh_stub import MeshStub, completion, text_stream
from tests.unit.test_modelmesh_client import GROQ, FakeTime, direct, mesh
from tests.unit.test_research_graph import CHUNKS, PAGES, Clock, analysis, answer_citing, coverage, good_draft, state

SCHEMAS = {m.__name__: m for m in (QueryAnalysis, ResearchPlan, Sufficiency, Reformulation, Draft, JudgeOutput)}


def as_request(body: dict[str, Any]) -> LLMRequest:
    metadata = body["metadata"]
    schema = SCHEMAS[body["response_format"]["json_schema"]["name"]] if "response_format" in body else None
    return LLMRequest(
        task_type=metadata.get("task_type_intended") or metadata["task_type"],
        messages=[ChatMessage(m["role"], m["content"]) for m in body["messages"]],
        correlation=Correlation(metadata["correlation_id"], metadata["workflow_node"], metadata["prompt_version"]),
        data_policy=body["routing"]["data_policy"] if "routing" in body else "any",
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        response_schema=schema,
    )


def scripted_modelmesh(llm: ScriptedLLM) -> Callable[[dict[str, Any]], httpx.Response]:
    def respond(body: dict[str, Any]) -> httpx.Response:
        request = as_request(body)
        if body.get("stream"):
            return text_stream(str(llm._next(request)))
        return completion(llm.complete(request).text)

    return respond


def scripted_groq(llm: ScriptedLLM) -> Callable[[httpx.Request], httpx.Response]:
    """The direct fallback: Groq's OpenAI format. Only the task type is unknown on the wire, so it is inferred."""

    def respond(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        schema = SCHEMAS[body["response_format"]["json_schema"]["name"]] if "response_format" in body else None
        task = {
            "QueryAnalysis": "query_classification",
            "Sufficiency": "evidence_assessment",
            "Draft": "evidence_synthesis",
            None: "final_response",
        }[schema.__name__ if schema else None]
        request = LLMRequest(
            task_type=task,  # type: ignore[arg-type]
            messages=[ChatMessage(m["role"], m["content"]) for m in body["messages"]],
            correlation=Correlation("run", "node", "p@1"),
            data_policy="any",
            max_tokens=100,
            response_schema=schema,
        )
        if body.get("stream"):
            text = str(llm._next(request))
            events = [{"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}]
            lines = [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]
            return httpx.Response(
                200, content="\n\n".join(lines).encode(), headers={"content-type": "text/event-stream"}
            )
        result = llm.complete(request)
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"content": result.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    return respond


def script() -> ScriptedLLM:
    return ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )


@pytest.fixture
def router() -> Iterator[respx.Router]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


def run_graph(client: Any, recorder: ListRecorder) -> Any:
    deps = WorkflowDeps(llm=client, search=KeywordSearch(CHUNKS), pages=PAGES, clock=Clock(), recorder=recorder)
    return run_to_end(build_graph(deps), state(), lambda _: None)


def test_full_research_run_through_modelmesh(router: respx.Router) -> None:
    stub = MeshStub(router)
    llm = script()
    stub.chat = [scripted_modelmesh(llm)] * 10
    recorder = ListRecorder()
    final = run_graph(mesh(FakeTime()), recorder)
    assert final.status == "completed"
    assert final.answer is not None and final.answer.rendering == "model"
    calls = final.llm_calls
    assert [c.task_type for c in calls] == [
        "query_classification",
        "evidence_assessment",
        "evidence_synthesis",
        "final_response",
    ]
    assert {c.backend for c in calls} == {"modelmesh"}
    # C3: every call joins run -> X-Request-ID -> mmr id -> trace id, and ModelMesh saw the run id.
    assert all(c.modelmesh_request_id and c.modelmesh_request_id.startswith("mmr_") for c in calls)
    assert all(c.modelmesh_trace_id for c in calls)
    assert [h["x-request-id"] for h in stub.headers] == [c.client_request_id for c in calls]
    assert {b["metadata"]["correlation_id"] for b in stub.bodies} == {final.run_id}
    # The evidence-bearing calls carry a fingerprint over what they showed; every call carries the namespace.
    assert len({b["cache"]["namespace"] for b in stub.bodies}) == 1
    fingerprints = [b["cache"].get("context_fingerprint") for b in stub.bodies]
    assert len(set(fingerprints)) == len(fingerprints)
    assert recorder.rows == calls
    summary = summarize(final)
    assert summary["backend"] == "modelmesh"
    assert summary["fallback"] is False


def test_modelmesh_stopping_mid_run_falls_back_to_groq_with_badges(router: respx.Router) -> None:
    stub = MeshStub(router)
    llm = script()
    # Two calls succeed, then ModelMesh is gone: connection refused for everything after.
    stub.chat = [scripted_modelmesh(llm)] * 2 + [httpx.ConnectError("refused")] * 20
    groq = router.post(GROQ).mock(side_effect=scripted_groq(llm))
    fake = FakeTime()
    final = run_graph(mesh(fake, fallback=direct(fake)), ListRecorder())
    assert final.status == "completed"
    assert final.answer is not None
    served = [(c.task_type, c.backend, c.status) for c in final.llm_calls if c.status == "ok"]
    assert served == [
        ("query_classification", "modelmesh", "ok"),
        ("evidence_assessment", "modelmesh", "ok"),
        ("evidence_synthesis", "direct", "ok"),
        ("final_response", "direct", "ok"),
    ]
    assert groq.call_count == 2
    fallback_rows = [c for c in final.llm_calls if c.backend == "direct"]
    assert {c.fallback_reason for c in fallback_rows} == {"modelmesh_unreachable"}
    summary = summarize(final)
    assert summary["backend"] == "modelmesh+direct"
    assert summary["backends"] == ["modelmesh", "direct"]
    assert summary["fallback"] is True
    assert summary["fallback_reasons"] == ["modelmesh_unreachable"]


def test_private_run_with_modelmesh_down_and_no_local_model_fails_clearly(router: respx.Router) -> None:
    stub = MeshStub(router)
    stub.chat = [httpx.ConnectError("refused")] * 4
    groq = router.post(GROQ).mock(side_effect=AssertionError("Groq must never see a private workspace"))
    fake = FakeTime()
    deps = WorkflowDeps(
        llm=mesh(fake, fallback=direct(fake, ollama=False)), search=KeywordSearch(CHUNKS), pages=PAGES, clock=Clock()
    )
    private = state().model_copy(update={"data_policy": "local_only"})
    final = run_to_end(build_graph(deps), private, lambda _: None)
    assert final.status == "failed"
    assert final.error is not None and final.error.code == "inference_not_permitted"
    assert {b["routing"]["data_policy"] for b in stub.bodies} == {"local_only"}
    assert not groq.called
