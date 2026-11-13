"""The `/runs` API with SSE progress, run against the real graph with a scripted LLM and an in-memory store."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mezvuat.api.deps import get_run_service, workspace_data_policy
from mezvuat.api.runs import RunEventBus, RunLimits, RunService
from mezvuat.core.clock import utcnow
from mezvuat.inference import InferenceUnavailableError
from mezvuat.orchestration import WorkflowDeps, memory_checkpointer
from tests.fakes import KeywordSearch, ListRecorder, ScriptedLLM
from tests.unit.test_research_graph import CHUNKS, PAGES, analysis, answer_citing, coverage, good_draft


class MemoryRunStore:
    def __init__(self, recorder: ListRecorder) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.recorder = recorder

    def create(self, *, run_id: str, workspace_id: str, question: str, data_policy: str, retrieval_mode: str) -> None:
        self.rows[run_id] = {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "question": question,
            "status": "queued",
            "data_policy": data_policy,
            "retrieval_mode": retrieval_mode,
            "created_at": utcnow(),
            "started_at": None,
            "finished_at": None,
            "error_code": None,
            "error_message": None,
            "answer": None,
            "summary": {},
        }

    def mark_running(self, run_id: str) -> None:
        self.rows[run_id].update(status="running", started_at=utcnow())

    def finish(self, run_id: str, **values: Any) -> None:
        self.rows[run_id].update(values, finished_at=utcnow())

    def get(self, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        row = self.rows.get(run_id)
        if row is None or row["workspace_id"] != workspace_id:
            return None
        calls = [
            {
                "client_request_id": c.client_request_id,
                "workflow_node": c.workflow_node,
                "task_type": c.task_type,
                "backend": c.backend,
                "provider": c.provider,
                "model": c.model,
                "status": c.status,
                "error_code": c.error_code,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "latency_ms": c.latency_ms,
            }
            for c in self.recorder.rows
            if c.run_id == run_id
        ]
        return {**row, "llm_calls": calls}


def happy_script() -> dict[str, list[Any]]:
    return {
        "query_classification": [analysis(single_hop=True)],
        "evidence_assessment": [coverage(True)],
        "evidence_synthesis": [good_draft],
        "final_response": [answer_citing("BM25 scores documents")],
    }


@pytest.fixture
def harness() -> Iterator[dict[str, Any]]:
    from mezvuat.api.app import create_app

    state: dict[str, Any] = {"llm": ScriptedLLM(happy_script()), "policies": [], "search": KeywordSearch(CHUNKS)}
    recorder = ListRecorder()

    @contextmanager
    def workflow(run_id: str, workspace_id: str, mode: str) -> Iterator[WorkflowDeps]:
        state["modes"] = [*state.get("modes", []), mode]
        yield WorkflowDeps(llm=state["llm"], search=state["search"], pages=PAGES, recorder=recorder)

    service = RunService(
        store=MemoryRunStore(recorder),
        workflow=workflow,
        checkpointer=memory_checkpointer,
        limits=RunLimits(deadline_s=60.0, max_llm_calls=12, max_tokens=100_000),
        concurrency=1,
        bus=RunEventBus(),
    )
    app = create_app()
    app.dependency_overrides[get_run_service] = lambda: service

    def policy(workspace_id: str) -> str:
        if workspace_id == "ws_missing":
            from mezvuat.core.errors import NotFoundError

            raise NotFoundError("workspace not found")
        return "any"

    app.dependency_overrides[workspace_data_policy] = policy
    with TestClient(app) as client:
        state.update(client=client, service=service, recorder=recorder)
        yield state
    service.shutdown()


def read_sse(client: TestClient, url: str, headers: dict[str, str] | None = None) -> list[tuple[int, str, Any]]:
    events: list[tuple[int, str, Any]] = []
    with client.stream("GET", url, headers=headers or {}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        current: dict[str, str] = {}
        for line in response.iter_lines():
            if line.startswith(":"):
                continue
            if not line:
                if current:
                    events.append((int(current["id"]), current["event"], json.loads(current["data"])))
                    current = {}
                continue
            key, _, value = line.partition(": ")
            current[key] = value
    return events


def wait_done(client: TestClient, url: str) -> dict[str, Any]:
    for _ in range(200):
        body = client.get(url).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_run_streams_progress_and_answer_then_persists_the_result(harness: dict[str, Any]) -> None:
    client: TestClient = harness["client"]
    created = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "How does BM25 rank documents?"})
    assert created.status_code == 202
    body = created.json()
    assert body["run_id"].startswith("run_")
    events = read_sse(client, body["events_url"])
    kinds = [kind for _, kind, _ in events]
    assert kinds[0] == "run_queued" and kinds[-1] == "run_completed"
    nodes = [data["node"] for _, kind, data in events if kind == "node"]
    assert nodes == [
        "understand_query",
        "plan_research",
        "retrieve",
        "assess_evidence",
        "synthesize",
        "verify",
        "respond",
    ]
    streamed = "".join(data["text"] for _, kind, data in events if kind == "answer_delta")
    final = events[-1][2]["answer"]
    assert streamed.strip() and streamed.strip() in final["markdown"]
    assert final["citations"] and all(c["page"] in (1, 2) for c in final["citations"])
    assert [i for i, _, _ in events] == list(range(1, len(events) + 1))

    run = wait_done(client, body["run_url"])
    assert run["status"] == "completed"
    assert run["data_policy"] == "any" and run["retrieval_mode"] == "hybrid_rrf"
    assert run["answer"] == final
    assert [c["task_type"] for c in run["llm_calls"]] == [
        "query_classification",
        "evidence_assessment",
        "evidence_synthesis",
        "final_response",
    ]
    assert run["summary"]["llm_calls"] == 4 and run["summary"]["backend"] == "direct"
    assert harness["modes"] == ["hybrid_rrf"]


def test_as_of_is_passed_to_the_run(harness: dict[str, Any]) -> None:
    client: TestClient = harness["client"]
    body = client.post(
        "/api/v1/workspaces/ws_1/runs", json={"question": "How does BM25 rank documents?", "as_of": "2024-06-01"}
    ).json()
    run = wait_done(client, body["run_url"])
    assert run["summary"]["as_of"] == "2024-06-01"


def test_reconnecting_with_last_event_id_resumes_after_it(harness: dict[str, Any]) -> None:
    client: TestClient = harness["client"]
    body = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "How does BM25 rank documents?"}).json()
    full = read_sse(client, body["events_url"])
    resumed = read_sse(client, body["events_url"], headers={"Last-Event-ID": "3"})
    assert [e[0] for e in resumed] == [e[0] for e in full if e[0] > 3]


def test_failed_run_reports_the_error(harness: dict[str, Any]) -> None:
    harness["llm"] = ScriptedLLM({"query_classification": [InferenceUnavailableError("provider down")]})
    client: TestClient = harness["client"]
    body = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "q?"}).json()
    events = read_sse(client, body["events_url"])
    assert events[-1][1] == "run_failed"
    assert events[-1][2]["code"] == "inference_unavailable"
    run = wait_done(client, body["run_url"])
    assert run["status"] == "failed"
    assert run["error"]["code"] == "inference_unavailable"
    assert run["answer"] is None
    assert len(run["llm_calls"]) == 1


def test_retrieval_outage_fails_the_run_without_an_answer(harness: dict[str, Any]) -> None:
    harness["llm"] = ScriptedLLM({"query_classification": [analysis(single_hop=True)]})
    harness["search"] = KeywordSearch(CHUNKS, fail=True)
    client: TestClient = harness["client"]
    body = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "q?"}).json()
    run = wait_done(client, body["run_url"])
    assert run["status"] == "failed" and run["error"]["code"] == "retrieval_unavailable"


def test_crash_inside_the_workflow_is_recorded_as_failed(harness: dict[str, Any]) -> None:
    class Exploding(KeywordSearch):
        def search(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("bug")

    harness["llm"] = ScriptedLLM({"query_classification": [analysis(single_hop=True)]})
    harness["search"] = Exploding(CHUNKS)
    client: TestClient = harness["client"]
    body = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "q?"}).json()
    events = read_sse(client, body["events_url"])
    assert events[-1][1] == "run_failed" and events[-1][2]["code"] == "internal_error"
    assert "bug" not in json.dumps(events[-1][2])


def test_unknown_run_and_workspace_are_404(harness: dict[str, Any]) -> None:
    client: TestClient = harness["client"]
    assert client.get("/api/v1/workspaces/ws_1/runs/run_nope").status_code == 404
    assert client.post("/api/v1/workspaces/ws_missing/runs", json={"question": "q"}).status_code == 404
    assert client.post("/api/v1/workspaces/ws_1/runs", json={"question": ""}).status_code == 422


def test_run_from_another_workspace_is_not_visible(harness: dict[str, Any]) -> None:
    client: TestClient = harness["client"]
    body = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "How does BM25 rank documents?"}).json()
    wait_done(client, body["run_url"])
    assert client.get(f"/api/v1/workspaces/ws_2/runs/{body['run_id']}").status_code == 404
    assert client.get(f"/api/v1/workspaces/ws_2/runs/{body['run_id']}/events").status_code == 404


def test_events_for_a_run_no_longer_in_memory_come_from_the_store(harness: dict[str, Any]) -> None:
    client: TestClient = harness["client"]
    service: RunService = harness["service"]
    body = client.post("/api/v1/workspaces/ws_1/runs", json={"question": "How does BM25 rank documents?"}).json()
    wait_done(client, body["run_url"])
    service.bus = RunEventBus()  # as after an API restart
    events = read_sse(client, body["events_url"])
    assert len(events) == 1 and events[0][1] == "run_completed"
    assert events[0][2]["answer"]["citations"]
