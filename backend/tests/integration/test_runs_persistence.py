"""Run persistence against the compose Postgres: run rows, llm_call rows, page maps and the LangGraph checkpointer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mezvuat.citations import PageSpan
from mezvuat.core.config import get_settings
from mezvuat.core.ids import new_id
from mezvuat.orchestration import (
    Budget,
    ResearchState,
    WorkflowDeps,
    build_graph,
    checkpoint_serializer,
    final_state,
    run_to_end,
)
from mezvuat.persistence.db import get_session_factory
from mezvuat.persistence.models import Document, DocumentVersion, Workspace
from mezvuat.persistence.runs import PostgresPageMap, SqlCallRecorder, SqlRunStore, postgres_checkpointer
from tests.fakes import KeywordSearch, ScriptedLLM
from tests.unit.test_research_graph import CHUNKS, analysis, answer_citing, coverage, good_draft

pytestmark = pytest.mark.integration


def _workspace() -> str:
    sessions = get_session_factory()
    workspace_id = new_id("ws")
    with sessions.begin() as session:
        session.add(Workspace(id=workspace_id, name="runs", data_classification="public"))
        session.flush()
        session.add(Document(id="doc_p", workspace_id=workspace_id))
        session.flush()
        session.add(
            DocumentVersion(
                id="ver_p",
                document_id="doc_p",
                workspace_id=workspace_id,
                source_type="upload",
                pages=[{"page": 1, "start": 0, "end": 150}, {"page": 2, "start": 150, "end": 400}],
            )
        )
    return workspace_id


def test_run_rows_calls_pages_and_checkpoints_round_trip(clean_db: None) -> None:
    sessions = get_session_factory()
    workspace_id = _workspace()
    store = SqlRunStore(sessions)
    run_id = new_id("run")
    store.create(run_id=run_id, workspace_id=workspace_id, question="q", data_policy="any", retrieval_mode="bm25")
    store.mark_running(run_id)

    pages = PostgresPageMap(sessions).page_spans(workspace_id, ["ver_p", "ver_missing"])
    assert pages == {"ver_p": [PageSpan(page=1, start=0, end=150), PageSpan(page=2, start=150, end=400)]}
    assert PostgresPageMap(sessions).page_spans("ws_other", ["ver_p"]) == {}

    search = KeywordSearch([c for c in CHUNKS if c.workspace_id == "ws_1"])
    llm = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [good_draft],
            "final_response": [answer_citing("BM25 scores documents")],
        }
    )
    saver = postgres_checkpointer(get_settings().psycopg_dsn, checkpoint_serializer(), max_size=2)
    graph = build_graph(
        WorkflowDeps(llm=llm, search=search, recorder=SqlCallRecorder(sessions, workspace_id)), checkpointer=saver
    )
    initial = ResearchState(
        run_id=run_id,
        workspace_id="ws_1",  # the fake search's workspace; the recorder writes rows under the real one
        question="How does BM25 rank documents?",
        data_policy="any",
        budget=Budget(llm_calls_left=12, tokens_left=100_000, deadline=1e12),
    )
    final = run_to_end(graph, initial)
    assert final.status == "completed"
    # The checkpoint in Postgres revives the same typed state.
    revived = final_state(graph, run_id)
    assert revived.answer == final.answer
    assert len(list(saver.list({"configurable": {"thread_id": run_id}}))) >= 8

    assert final.answer is not None
    store.finish(
        run_id,
        status="completed",
        answer=final.answer.model_dump(mode="json"),
        summary={"llm_calls": len(final.llm_calls)},
        error_code=None,
        error_message=None,
    )
    row = store.get(workspace_id, run_id)
    assert row is not None and row["status"] == "completed"
    assert row["answer"]["citations"]
    assert [c["task_type"] for c in row["llm_calls"]] == [r.task_type for r in final.llm_calls]
    assert store.get("ws_other", run_id) is None
    saver.conn.close()


def test_temporal_source_returns_typed_dates_and_never_the_ingestion_time(clean_db: None) -> None:
    from datetime import date

    from mezvuat.persistence.models import TemporalFact
    from mezvuat.persistence.runs import PostgresTemporalSource

    sessions = get_session_factory()
    workspace_id = _workspace()
    with sessions.begin() as session:
        session.add(
            TemporalFact(
                document_version_id="ver_p",
                kind="published",
                value=date(2023, 2, 1),
                precision="day",
                source="source_metadata",
                confidence="high",
                evidence="arXivRaw v1 date",
            )
        )
    versions = PostgresTemporalSource(sessions).versions(workspace_id, ["doc_p"])
    [version] = versions["doc_p"]
    assert version.version_id == "ver_p"
    assert [(f.kind, f.value, f.precision) for f in version.facts] == [("published", date(2023, 2, 1), "day")]
    assert PostgresTemporalSource(sessions).versions("ws_other", ["doc_p"]) == {}


def test_feedback_rows_are_scoped_to_the_run_workspace(clean_db: None) -> None:
    sessions = get_session_factory()
    workspace_id = _workspace()
    store = SqlRunStore(sessions)
    run_id = new_id("run")
    store.create(run_id=run_id, workspace_id=workspace_id, question="q", data_policy="any", retrieval_mode="bm25")
    row = store.add_feedback(workspace_id, run_id, rating="down", claim_id="C1", comment="wrong page")
    assert row is not None and row["feedback_id"].startswith("fb_") and row["claim_id"] == "C1"
    assert store.add_feedback("ws_other", run_id, rating="up", claim_id=None, comment=None) is None
    assert store.add_feedback(workspace_id, "run_missing", rating="up", claim_id=None, comment=None) is None


def test_version_text_endpoint_returns_the_page_holding_an_offset(client: TestClient) -> None:
    sessions = get_session_factory()
    workspace_id = _workspace()
    text = "a" * 150 + "Page two starts here. " + "b" * 228
    with sessions.begin() as session:
        version = session.get(DocumentVersion, "ver_p")
        assert version is not None
        version.normalized_text = text
    base = f"/api/v1/workspaces/{workspace_id}/documents/doc_p/versions/ver_p/text"
    body = client.get(base, params={"at": 155}).json()
    assert (body["page"], body["page_count"], body["start"], body["end"]) == (2, 2, 150, 400)
    assert body["text"].startswith("Page two starts here.")
    assert client.get(base, params={"page": 1}).json()["text"] == "a" * 150
    assert client.get(base, params={"page": 9}).status_code == 404
    other = "/api/v1/workspaces/ws_other/documents/doc_p/versions/ver_p/text"
    assert client.get(other).status_code == 404
