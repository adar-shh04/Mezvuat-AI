"""M6 evaluation scaffolding with fakes only: the custom metrics, RAGAS behind a scripted judge, the batch runner with
a fake tracker, and the ModelMesh labels it would queue. No real judge, no real evaluation (the labels are not
finished), no network."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import respx
from pydantic import BaseModel

from mezvuat.evaluation.answer_eval import (
    EvalResult,
    evaluate,
    feedback_items,
    load_temporal_gold,
    render_markdown,
    validate_items,
    write_outputs,
)
from mezvuat.evaluation.answer_metrics import answer_metrics, citation_validity
from mezvuat.evaluation.judge import LLMClientJudge
from mezvuat.inference import LLMRequest, portability_problems, portable_schema
from mezvuat.orchestration import ResearchState, WorkflowDeps, build_graph, run_to_end
from tests.fakes import KeywordSearch, ScriptedLLM
from tests.modelmesh_stub import MeshStub, assert_valid, completion, mmr_id
from tests.unit.test_modelmesh_client import FakeTime, mesh
from tests.unit.test_research_graph import CHUNKS, PAGES, Clock, analysis, answer_citing, coverage, good_draft, state

ragas = pytest.importorskip("ragas", reason="the eval dependency group is not installed")

from ragas.metrics.collections.answer_relevancy.util import AnswerRelevanceOutput  # noqa: E402
from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput  # noqa: E402
from ragas.metrics.collections.context_recall.util import ContextRecallOutput  # noqa: E402
from ragas.metrics.collections.faithfulness.util import NLIStatementOutput, StatementGeneratorOutput  # noqa: E402

from mezvuat.evaluation.ragas_metrics import RagasScorer  # noqa: E402

RAGAS_MODELS: list[type[BaseModel]] = [
    StatementGeneratorOutput,
    NLIStatementOutput,
    AnswerRelevanceOutput,
    ContextPrecisionOutput,
    ContextRecallOutput,
]
CANNED: dict[str, dict[str, Any]] = {
    "StatementGeneratorOutput": {"statements": ["BM25 scores documents by term frequency."]},
    "NLIStatementOutput": {
        "statements": [{"statement": "BM25 scores documents by term frequency.", "reason": "stated", "verdict": 1}]
    },
    "AnswerRelevanceOutput": {"question": "How does BM25 rank documents?", "noncommittal": 0},
    "ContextPrecisionOutput": {"reason": "useful", "verdict": 1},
    "ContextRecallOutput": {
        "classifications": [{"statement": "BM25 uses tf-idf", "reason": "in context", "attributed": 1}]
    },
}


def scripted_judge() -> ScriptedLLM:
    def answer(request: LLMRequest) -> dict[str, Any]:
        assert request.response_schema is not None
        return CANNED[request.response_schema.__name__]

    return ScriptedLLM({}, default={"evaluation_judge": answer})


def generator() -> ScriptedLLM:
    return ScriptedLLM(
        {},
        default={
            "query_classification": analysis(single_hop=True),
            "evidence_assessment": coverage(True),
            "evidence_synthesis": good_draft,
            "final_response": answer_citing("BM25 scores documents"),
        },
    )


def run(question: str = "How does BM25 rank documents?", llm: Any = None) -> ResearchState:
    deps = WorkflowDeps(llm=llm or generator(), search=KeywordSearch(CHUNKS), pages=PAGES, clock=Clock())
    return run_to_end(build_graph(deps), state(question=question))


def embed(text: str) -> list[float]:
    return [1.0, float(len(text) % 7), 0.5]


# ----- custom metrics -------------------------------------------------------------------------------------------------


def test_custom_metrics_on_a_verified_answer() -> None:
    final = run()
    item = {
        "item_id": "q1",
        "question": final.question,
        "expected_behavior": "answer",
        "gold_chunk_ids": ["c_bm25", "c_rrf"],
        "expected_versions": ["v1"],
    }
    metrics = answer_metrics(final, item)
    assert metrics["completed"] == 1.0
    assert metrics["citation_validity"] == 1.0
    assert metrics["abstention_correct"] == 1.0
    assert metrics["temporal_accuracy"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 0.5
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["llm_calls"] == 4
    assert "cost_estimated_usd" not in metrics  # the direct path reports no cost, and none is invented


def test_abstention_expected_and_wrong_versions() -> None:
    final = run()
    assert answer_metrics(final, {"item_id": "q", "expected_behavior": "abstain"})["abstention_correct"] == 0.0
    assert answer_metrics(final, {"item_id": "q", "expected_versions": ["v2"]})["temporal_accuracy"] == 0.0
    # Metrics that do not apply are absent, not zero.
    assert "citation_precision" not in answer_metrics(final, {"item_id": "q"})


def test_a_tampered_citation_fails_validity() -> None:
    final = run()
    assert final.answer is not None
    bad = final.answer.citations[0].model_copy(update={"quote": "a quote that is not in the evidence at all"})
    tampered = final.model_copy(update={"answer": final.answer.model_copy(update={"citations": [bad]})})
    assert citation_validity(tampered) == 0.0


# ----- the judge and RAGAS --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", RAGAS_MODELS, ids=lambda m: m.__name__)
def test_ragas_response_models_fit_the_portable_schema_subset(model: type[BaseModel]) -> None:
    """Judge calls use strict JSON schema on ModelMesh and Groq, so RAGAS's output models must be portable."""
    assert portability_problems(portable_schema(model)) == []


def test_ragas_metrics_run_behind_the_pinned_judge() -> None:
    llm = scripted_judge()
    scorer = RagasScorer(LLMClientJudge(llm, model="groq/qwen/qwen3.8-27b", eval_run_id="t1"), embed, strictness=1)
    contexts = ["BM25 is a bag-of-words ranking function that scores documents by term frequency."]
    scores = scorer.score(
        "How does BM25 rank documents?", "BM25 scores documents by term frequency.", contexts, "By tf-idf."
    )
    assert set(scores) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert scores["faithfulness"] == 1.0
    assert scores["context_precision"] == pytest.approx(1.0)
    assert scores["context_recall"] == 1.0
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # Every RAGAS call went through the judge as an evaluation_judge request with the pinned model.
    assert llm.requests and {r.task_type for r in llm.requests} == {"evaluation_judge"}
    assert {r.model for r in llm.requests} == {"groq/qwen/qwen3.8-27b"}
    assert {r.data_policy for r in llm.requests} == {"any"}
    assert {r.temperature for r in llm.requests} == {0.0}


def test_missing_inputs_skip_metrics_instead_of_scoring_zero() -> None:
    scorer = RagasScorer(LLMClientJudge(scripted_judge(), model="m", eval_run_id="t"), embed, strictness=1)
    assert scorer.score("q?", None, [], None) == {}
    assert set(scorer.score("q?", "an answer", [], None)) == {"answer_relevancy"}


@pytest.fixture
def router() -> Iterator[respx.Router]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


def test_judge_through_modelmesh_is_pinned_with_fallback_none(router: respx.Router) -> None:
    stub = MeshStub(router)
    stub.chat = [completion(json.dumps(CANNED["ContextPrecisionOutput"]))]
    judge = LLMClientJudge(mesh(FakeTime()), model="groq/qwen/qwen3.8-27b", eval_run_id="abc")
    out = judge.structured("Is the context useful?", ContextPrecisionOutput)
    assert out.verdict == 1
    body = stub.bodies[0]
    assert body["model"] == "groq/qwen/qwen3.8-27b"
    assert body["metadata"]["task_type"] == "evaluation_judge"
    assert body["metadata"]["correlation_id"] == "eval_abc"
    assert body["routing"]["fallback"] == "none"
    assert body["routing"]["data_policy"] == "any"
    assert "min_quality" not in body["routing"]
    assert body["cache"]["mode"] == "exact"


# ----- the batch runner -----------------------------------------------------------------------------------------------


class FakeTracker:
    def __init__(self) -> None:
        self.runs: list[tuple[str, bool]] = []
        self.params: list[Mapping[str, Any]] = []
        self.metrics: list[Mapping[str, float]] = []
        self.artifacts: dict[str, Any] = {}

    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[None]:
        self.runs.append((name, nested))
        yield

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.metrics.append(metrics)

    def log_json(self, data: Any, name: str) -> None:
        self.artifacts[name] = data

    def run_id(self) -> str:
        return "0123abcd"


class FixedScorer:
    def score(self, question: str, answer: str | None, contexts: Any, reference: str | None) -> dict[str, float]:
        return {"faithfulness": 0.9, "answer_relevancy": 0.8} if answer else {}


ITEMS = [
    {
        "item_id": "q1",
        "question": "How does BM25 rank documents?",
        "expected_behavior": "answer",
        "reference_answer": "tf-idf",
    },
    {"item_id": "q2", "question": "How does BM25 rank documents?", "expected_behavior": "abstain"},
]


def test_evaluate_runs_every_config_and_logs_to_the_tracker(tmp_path: Path) -> None:
    tracker = FakeTracker()
    configs_seen: list[str] = []

    def run_item(config: str, item: Mapping[str, Any], run_id: str) -> ResearchState:
        configs_seen.append(config)
        return run(str(item["question"])).model_copy(update={"run_id": run_id})

    result = evaluate(ITEMS, ["bm25", "hybrid_rrf"], run_item, tracker, scorer=FixedScorer(), judge_model="judge-x")
    assert configs_seen == ["bm25", "bm25", "hybrid_rrf", "hybrid_rrf"]
    assert tracker.runs == [("answer-eval", False), ("bm25", True), ("hybrid_rrf", True)]
    assert result.eval_run_id == "mlflow:0123abcd"
    summary = result.summary()
    bm25 = summary["per_config"]["bm25"]["metrics"]
    assert bm25["abstention_correct"] == {"mean": 0.5, "n": 2}
    assert bm25["faithfulness"] == {"mean": 0.9, "n": 2}
    assert bm25["citation_validity"]["mean"] == 1.0
    assert "rows_bm25.json" in tracker.artifacts and "langfuse_traces_bm25.json" in tracker.artifacts
    assert {"faithfulness_mean", "citation_validity_mean"} <= set(tracker.metrics[0])
    report = write_outputs(tmp_path, result).read_text(encoding="utf-8")
    assert report == render_markdown(summary)
    assert "| bm25 | 2 |" in report


def test_unknown_configs_and_unreviewed_items_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown retrieval"):
        evaluate(ITEMS, ["bm42"], lambda *_: run(), FakeTracker())
    with pytest.raises(ValueError, match="D11"):
        validate_items([{"item_id": "a", "question": "q", "status": "draft"}])
    with pytest.raises(ValueError, match="duplicate"):
        validate_items([{"item_id": "a", "question": "q"}, {"item_id": "a", "question": "q"}])


def test_feedback_labels_for_modelmesh_requests_follow_the_contract() -> None:
    synth, final = mmr_id(), mmr_id()
    result = EvalResult(eval_run_id="mlflow:0123abcd", dataset_version="gold-abc", judge_model="judge-x")
    result.rows.append(
        {
            "item_id": "q1",
            "config": "bm25",
            "metrics": {
                "completed": 1.0,
                "abstention_correct": 1.0,
                "citation_validity": 1.0,
                "faithfulness": 0.9,
                "answer_relevancy": 0.7,
            },
            "calls": [
                {"task_type": "evidence_synthesis", "status": "ok", "modelmesh_request_id": synth},
                {"task_type": "final_response", "status": "ok", "modelmesh_request_id": final},
                {"task_type": "final_response", "status": "error", "modelmesh_request_id": mmr_id()},
                {"task_type": "query_classification", "status": "ok", "modelmesh_request_id": None},
            ],
        }
    )
    labels = feedback_items(
        result,
        evaluator="mezvuat-eval/ragas-0.4.4+judge:judge-x",
        ragas_metric_version="ragas-0.4.4",
        labelled_at=datetime(2026, 11, 17, tzinfo=UTC),
    )
    wire = [label.wire() for label in labels]
    for item in wire:
        assert_valid("FeedbackRequest", item)
        assert item["eval_run_id"] == "mlflow:0123abcd" and item["metric_version"] and item["dataset_version"]
    direct = {(w["metric"], w["request_id"]) for w in wire if w["attribution"] == "direct"}
    assert direct == {("faithfulness", synth), ("answer_relevancy", final)}
    workflow = [w for w in wire if w["attribution"] == "workflow"]
    assert {w["request_id"] for w in workflow} == {synth, final}
    assert {w["metric"] for w in workflow} == {"task_success"}
    assert all(w["score"] == 1.0 and w["accepted"] is True for w in workflow)


def test_temporal_gold_merges_reviewed_decisions_only(tmp_path: Path) -> None:
    drafts = [
        {
            "item_id": f"p:{k}",
            "question": f"Q {k}?",
            "expected_behavior": "answer",
            "expected_versions": ["v1"],
            "as_of": None,
        }
        for k in ("a", "b", "c", "d")
    ]
    review = [
        {"item_id": "p:a", "decision": "accept", "answer": "A.", "expected_behavior": "answer"},
        {"item_id": "p:b", "decision": "reject"},
        {"item_id": "p:c", "decision": "write", "answer": "C.", "expected_behavior": "abstain"},
        {"item_id": "p:c", "retracted": True},
        {"item_id": "p:d", "decision": "edit", "answer": "D!", "expected_behavior": "abstain"},
    ]
    (tmp_path / "drafts.jsonl").write_text("\n".join(json.dumps(d) for d in drafts), encoding="utf-8")
    (tmp_path / "review.human.jsonl").write_text("\n".join(json.dumps(r) for r in review), encoding="utf-8")
    gold = load_temporal_gold(tmp_path)
    assert [(g["item_id"], g["reference_answer"], g["expected_behavior"]) for g in gold] == [
        ("p:a", "A.", "answer"),
        ("p:d", "D!", "abstain"),
    ]
    validate_items(gold)
