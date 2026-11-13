"""Run persistence against the compose Postgres: run rows, llm_call rows, page maps and the LangGraph checkpointer."""

from __future__ import annotations

import pytest

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
