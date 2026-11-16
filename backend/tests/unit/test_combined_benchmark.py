"""The combined benchmark's plumbing with fakes only (M8, C7). The benchmark itself has not been run.

The direct arm uses the scripted LLM; the ModelMesh arm goes through the real client against the mocked ModelMesh,
and cost and latency come from mocked `GET /v1/requests/{id}` records that validate against the spec.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import respx

from mezvuat.evaluation.combined_benchmark import Arm, read_items, render_markdown, run_benchmark, write_outputs
from mezvuat.orchestration import ResearchState, WorkflowDeps, build_graph, run_to_end
from tests.fakes import KeywordSearch, ScriptedLLM
from tests.modelmesh_stub import MeshStub, assert_valid, modelmesh_block
from tests.unit.test_modelmesh_client import FakeTime, mesh
from tests.unit.test_modelmesh_workflow import scripted_modelmesh
from tests.unit.test_research_graph import CHUNKS, PAGES, Clock, analysis, answer_citing, coverage, good_draft, state


def script() -> ScriptedLLM:
    return ScriptedLLM(
        {},
        default={
            "query_classification": analysis(single_hop=True),
            "evidence_assessment": coverage(True),
            "evidence_synthesis": good_draft,
            "final_response": answer_citing("BM25 scores documents"),
        },
    )


def record(request_id: str) -> dict[str, Any]:
    block = modelmesh_block(request_id=request_id, cost=0.001)
    rec = {
        "id": request_id,
        "client_request_id": "llmreq_x",
        "created_at": "2026-11-17T12:00:00Z",
        "completed_at": "2026-11-17T12:00:01Z",
        "status": "succeeded",
        "http_status": 200,
        "api_key_id": "key_1",
        "endpoint": "chat.completions",
        "stream": False,
        "metadata": {"application": "mezvuat"},
        "modelmesh": block,
        "error": None,
        "feedback_count": 0,
        "content": None,
    }
    assert_valid("RequestRecord", rec)
    return rec


@pytest.fixture
def router() -> Iterator[respx.Router]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


def test_arms_parse() -> None:
    assert Arm.parse("direct") == Arm("direct", "direct")
    assert Arm.parse("modelmesh:cost_optimized").policy == "cost_optimized"
    with pytest.raises(ValueError):
        Arm.parse("modelmesh:cheapest")


def test_benchmark_rows_and_report_come_from_data(router: respx.Router, tmp_path: Path) -> None:
    stub = MeshStub(router)
    mesh_llm = script()
    stub.chat = [scripted_modelmesh(mesh_llm)] * 20
    clients: dict[str, Any] = {"direct": script(), "modelmesh:cost_optimized": mesh(FakeTime())}
    clients["modelmesh:cost_optimized"].backend.policies = {"query_classification": "cost_optimized"}

    def run_item(arm: Arm, item: Mapping[str, Any], run_id: str) -> ResearchState:
        deps = WorkflowDeps(llm=clients[arm.name], search=KeywordSearch(CHUNKS), pages=PAGES, clock=Clock())
        initial = state(question=str(item["question"])).model_copy(update={"run_id": run_id})
        return run_to_end(build_graph(deps), initial)

    items = [
        {"item_id": "q1", "question": "How does BM25 rank documents?", "expected_behavior": "answer"},
        {"item_id": "q2", "question": "How does BM25 rank documents?", "expected_behavior": "cannot_determine"},
    ]
    arms = [Arm.parse("direct"), Arm.parse("modelmesh:cost_optimized")]
    result = run_benchmark(items, arms, run_item, record)
    assert [(r["item_id"], r["arm"]) for r in result.rows] == [
        ("q1", "direct"),
        ("q1", "modelmesh:cost_optimized"),
        ("q2", "direct"),
        ("q2", "modelmesh:cost_optimized"),
    ]
    assert stub.bodies[0]["routing"]["policy"] == "cost_optimized"
    summary = result.summary()
    direct, routed = summary["per_arm"]["direct"], summary["per_arm"]["modelmesh:cost_optimized"]
    assert direct["cost_estimated_usd_total"] is None  # the direct backend reports no cost; nothing is estimated
    # Four ModelMesh calls per run, each record says 0.001 USD and 760 ms.
    assert routed["cost_estimated_usd_total"] == pytest.approx(2 * 4 * 0.001)
    assert routed["llm_latency_ms_median"] == 4 * 760
    assert routed["metrics"]["abstention_correct"] == {"mean": 0.5, "n": 2}
    assert routed["records_missing"] == 0
    report = write_outputs(tmp_path, result, {"direct": "abc"})
    text = report.read_text(encoding="utf-8")
    assert "| modelmesh:cost_optimized | 2 |" in text
    assert "not reported" in text
    assert text == render_markdown(summary, mlflow_runs={"direct": "abc"})
    assert (tmp_path / "results.json").exists()


def test_only_reviewed_items_are_read(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"item_id": "a", "question": "q", "status": "draft"}\n'
        '{"item_id": "b", "question": "q", "status": "confirmed"}\n\n',
        encoding="utf-8",
    )
    assert [i["item_id"] for i in read_items(path)] == ["b"]
