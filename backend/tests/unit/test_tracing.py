"""Langfuse tracing (M6, decision D3): optional, masked before export, and never able to break a run.

The real Langfuse SDK runs here, with an in-memory span exporter instead of the network: what the exporter receives
is exactly what would leave the process.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mezvuat.observability.tracing import LangfuseTracer, NoopTracer, build_tracer, mask_content
from mezvuat.orchestration import WorkflowDeps, build_graph, run_to_end
from mezvuat.orchestration.prompts import load_prompt
from tests.fakes import KeywordSearch, ListRecorder, ScriptedLLM
from tests.unit.test_research_graph import (
    BM25,
    CHUNKS,
    DENSE,
    PAGES,
    Clock,
    analysis,
    answer_citing,
    coverage,
    good_draft,
    state,
)

HOST = "https://langfuse.test"
QUESTION = "How does BM25 rank documents?"


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
        # Langfuse's own API calls (scores, media) stay in the process too.
        mocked.route(host="langfuse.test").mock(return_value=httpx.Response(200, json={}))
        yield mocked


def tracer(key: str, exporter: SpanExporter, *, capture: bool = False) -> LangfuseTracer:
    built = build_tracer(
        public_key=key,
        secret_key="sk-test",
        host=HOST,
        capture_content=capture,
        span_exporter=exporter,
        httpx_client=httpx.Client(),
        flush_at=1,
    )
    assert isinstance(built, LangfuseTracer)
    return built


def traced_run(trace: LangfuseTracer, recorder: ListRecorder | None = None) -> Any:
    deps = WorkflowDeps(
        llm=script(),
        search=KeywordSearch(CHUNKS),
        pages=PAGES,
        clock=Clock(),
        recorder=recorder,
        langfuse_trace_id=trace.trace_id,
    )
    initial = state(question=QUESTION)
    with trace.run(initial.run_id, metadata={"workspace_id": initial.workspace_id}) as callbacks:
        assert callbacks, "the Langfuse callback handler should be active"
        final = run_to_end(build_graph(deps), initial, callbacks=callbacks)
    trace.flush()
    return final


def exported_text(spans: Sequence[ReadableSpan]) -> str:
    parts = []
    for span in spans:
        parts.append(span.name)
        parts.extend(f"{k}={v}" for k, v in (span.attributes or {}).items())
        parts.extend(json.dumps(dict(e.attributes or {}), default=str) for e in span.events)
    return "\n".join(parts)


def test_masking_keeps_ids_and_structure_and_hides_text() -> None:
    masked = mask_content(
        data={
            "run_id": "run_01JTESTRUN",
            "node": "synthesize",
            "question": QUESTION,
            "evidence": [{"id": "E1", "text": BM25}],
            "count": 3,
            "ok": True,
        }
    )
    assert masked["run_id"] == "run_01JTESTRUN"
    assert masked["node"] == "synthesize"
    assert masked["evidence"][0]["id"] == "E1"
    assert masked["count"] == 3 and masked["ok"] is True
    assert masked["question"].startswith("<masked 29 chars sha256:")
    assert BM25 not in json.dumps(masked)


def test_without_keys_tracing_is_a_no_op() -> None:
    assert isinstance(build_tracer(public_key=None, secret_key=None, host=None), NoopTracer)
    assert isinstance(build_tracer(public_key="pk", secret_key="", host=None), NoopTracer)
    noop = NoopTracer()
    with noop.run("run_1", metadata={}) as callbacks:
        assert callbacks == []
    assert noop.trace_id("run_1") is None


def test_no_prompt_or_document_text_reaches_the_exporter(router: respx.Router) -> None:
    exporter = InMemorySpanExporter()
    trace = tracer("pk-test-mask", exporter)
    recorder = ListRecorder()
    final = traced_run(trace, recorder)
    assert final.status == "completed"
    spans = exporter.get_finished_spans()
    assert len(spans) > 3, "the graph's node spans should have been exported"
    text = exported_text(spans)
    system_prompt = load_prompt("system").template
    for secret in (QUESTION, BM25[:60], DENSE[:60], system_prompt[:60], "term frequency and inverse document"):
        assert secret not in text
    assert "<masked " in text  # inputs and outputs were there, masked
    # One trace per run, with the id Mezvuat stores on every llm_call row.
    trace_id = trace.trace_id(final.run_id)
    assert trace_id is not None
    assert {format(s.context.trace_id, "032x") for s in spans} == {trace_id}
    assert recorder.rows and {r.langfuse_trace_id for r in recorder.rows} == {trace_id}


def test_capture_content_is_an_explicit_opt_in(router: respx.Router) -> None:
    exporter = InMemorySpanExporter()
    traced_run(tracer("pk-test-capture", exporter, capture=True))
    assert QUESTION in exported_text(exporter.get_finished_spans())


class FailingExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise ConnectionError("Langfuse is down")

    def shutdown(self) -> None:
        return None


def test_runs_succeed_when_langfuse_is_down(router: respx.Router) -> None:
    final = traced_run(tracer("pk-test-down", FailingExporter()))
    assert final.status == "completed"
    assert final.answer is not None


def test_runs_succeed_when_tracing_cannot_start() -> None:
    class Broken:
        def start_as_current_observation(self, **_: Any) -> Any:
            raise RuntimeError("SDK misconfigured")

    trace = LangfuseTracer(Broken())
    with trace.run("run_01JTESTRUN", metadata={}) as callbacks:
        assert callbacks == []
    trace.score("run_01JTESTRUN", "user_thumbs", 1.0)  # swallowed, logged
