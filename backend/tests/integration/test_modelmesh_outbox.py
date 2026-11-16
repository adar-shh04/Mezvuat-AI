"""The ModelMesh outbox against the compose Postgres (M8): workspace deletion queues the cache purge and the feedback
deletion, ratings queue feedback, and the drain sends them to a mocked ModelMesh in the contract's format."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select, text

from mezvuat.core.ids import new_id
from mezvuat.inference.modelmesh import ModelMeshOps, cache_namespace
from mezvuat.orchestration import LLMCallRecord
from mezvuat.persistence.db import get_engine, get_session_factory
from mezvuat.persistence.modelmesh_outbox import drain_outbox, enqueue_run_thumbs
from mezvuat.persistence.models import LlmCall, ModelMeshOutbox, Workspace
from mezvuat.persistence.runs import SqlCallRecorder, SqlRunStore
from mezvuat.workspaces import service
from tests.modelmesh_stub import BASE, assert_valid, mmr_id
from tests.unit.test_modelmesh_contract import batch_result

pytestmark = pytest.mark.integration


class _Nothing:
    def delete_workspace(self, workspace_id: str) -> None:
        return None

    def delete(self, digest: str) -> None:
        return None


@pytest.fixture
def outbox(clean_db: None) -> Iterator[None]:
    yield
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE modelmesh_outbox"))


@pytest.fixture
def router() -> Iterator[respx.Router]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


def _call(run_id: str, mmr: str | None, backend: str = "modelmesh") -> LLMCallRecord:
    return LLMCallRecord(
        client_request_id=new_id("llmreq"),
        run_id=run_id,
        workflow_node="synthesize",
        task_type="evidence_synthesis",
        backend=backend,
        provider="groq",
        model="groq/openai/gpt-oss-120b",
        prompt_version="synthesize@1",
        iteration=0,
        status="ok",
        error_code=None,
        http_status=200,
        provider_request_id=mmr,
        prompt_tokens=10,
        completion_tokens=5,
        usage_source="provider",
        latency_ms=100,
        finish_reason="stop",
        structured_mode="json_schema_strict",
        modelmesh_request_id=mmr,
        modelmesh_trace_id="4bf92f3577b34da6a3ce929d0e0e4736" if mmr else None,
        cache_status="miss" if mmr else None,
        cost_estimated_usd=0.0005 if mmr else None,
        fallback_reason=None if mmr else "modelmesh_503",
    )


def _workspace_with_calls() -> tuple[str, str, list[str]]:
    sessions = get_session_factory()
    workspace_id = new_id("ws")
    with sessions.begin() as session:
        session.add(Workspace(id=workspace_id, name="mm", data_classification="public"))
    run_id = new_id("run")
    SqlRunStore(sessions).create(
        run_id=run_id, workspace_id=workspace_id, question="q", data_policy="any", retrieval_mode="bm25"
    )
    ids = [mmr_id(), mmr_id()]
    SqlCallRecorder(sessions, workspace_id).record(
        [_call(run_id, ids[0]), _call(run_id, ids[1]), _call(run_id, None, "direct")]
    )
    return workspace_id, run_id, ids


def test_llm_call_rows_keep_the_modelmesh_correlation(outbox: None) -> None:
    workspace_id, run_id, ids = _workspace_with_calls()
    with get_session_factory()() as session:
        rows = session.scalars(select(LlmCall).where(LlmCall.run_id == run_id)).all()
    assert sorted(r.modelmesh_request_id for r in rows if r.modelmesh_request_id) == sorted(ids)
    direct = next(r for r in rows if r.backend == "direct")
    assert direct.fallback_reason == "modelmesh_503"
    assert {r.modelmesh_trace_id for r in rows if r.backend == "modelmesh"} == {"4bf92f3577b34da6a3ce929d0e0e4736"}
    run = SqlRunStore(get_session_factory()).get(workspace_id, run_id)
    assert run is not None
    assert {c["fallback_reason"] for c in run["llm_calls"]} == {None, "modelmesh_503"}


def test_workspace_delete_queues_and_drains_the_modelmesh_cleanup(outbox: None, router: respx.Router) -> None:
    workspace_id, _, ids = _workspace_with_calls()
    namespace = cache_namespace(b"secret", workspace_id)
    sessions = get_session_factory()
    with sessions() as session:
        service.delete_workspace(
            session,
            workspace_id,
            dense=_Nothing(),  # type: ignore[arg-type]
            lexical=_Nothing(),  # type: ignore[arg-type]
            blobs=_Nothing(),
            modelmesh_cleanup=True,
            cache_namespace=namespace,
        )
    with sessions() as session:
        rows = session.scalars(select(ModelMeshOutbox).order_by(ModelMeshOutbox.created_at)).all()
        assert sorted(r.kind for r in rows) == ["cache_invalidate", "feedback_delete", "feedback_delete"]
        assert session.get(Workspace, workspace_id) is None

    purges: list[dict[str, Any]] = []

    def purge(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        assert_valid("CacheInvalidateRequest", body)
        purges.append(body)
        return httpx.Response(200, json={"deleted_entries": 3})

    router.post(f"{BASE}/v1/cache/invalidate").mock(side_effect=purge)
    deletes = [
        router.delete(f"{BASE}/v1/feedback/{i}").mock(return_value=httpx.Response(200, json={"deleted": 1}))
        for i in ids
    ]
    ops = ModelMeshOps(base_url=BASE, api_key="k", http=httpx.Client(), sleep=lambda _: None)
    assert drain_outbox(sessions, ops) == 3
    assert purges == [{"namespace": namespace}]
    assert all(d.called for d in deletes)
    with sessions() as session:
        assert {r.status for r in session.scalars(select(ModelMeshOutbox))} == {"done"}


def test_drain_pauses_while_modelmesh_is_down(outbox: None, router: respx.Router) -> None:
    workspace_id, _, _ = _workspace_with_calls()
    sessions = get_session_factory()
    with sessions() as session:
        service.delete_workspace(
            session,
            workspace_id,
            dense=_Nothing(),  # type: ignore[arg-type]
            lexical=_Nothing(),  # type: ignore[arg-type]
            blobs=_Nothing(),
            modelmesh_cleanup=True,
            cache_namespace=cache_namespace(b"secret", workspace_id),
        )
    route = router.post(f"{BASE}/v1/cache/invalidate").mock(side_effect=httpx.ConnectError("refused"))
    ops = ModelMeshOps(base_url=BASE, http=httpx.Client(), sleep=lambda _: None, max_retries=0)
    assert drain_outbox(sessions, ops) == 0
    assert route.call_count == 1  # stopped at the first unavailable error
    with sessions() as session:
        rows = session.scalars(select(ModelMeshOutbox)).all()
        assert {r.status for r in rows} == {"queued"}
        assert sum(r.attempts for r in rows) == 1


def test_ratings_queue_user_thumbs_for_the_runs_modelmesh_requests(outbox: None, router: respx.Router) -> None:
    _, run_id, ids = _workspace_with_calls()
    sessions = get_session_factory()
    assert enqueue_run_thumbs(sessions, run_id, "up", datetime(2026, 11, 17, tzinfo=UTC)) == 2
    sent: list[dict[str, Any]] = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        assert_valid("FeedbackBatchRequest", body)
        sent.append(body)
        return batch_result(body)

    router.post(f"{BASE}/v1/feedback/batch").mock(side_effect=handle)
    ops = ModelMeshOps(base_url=BASE, http=httpx.Client(), sleep=lambda _: None)
    assert drain_outbox(sessions, ops) == 1
    items = sent[0]["items"]
    assert sorted(i["request_id"] for i in items) == sorted(ids)
    assert {i["attribution"] for i in items} == {"workflow"}
    assert {i["label_source"] for i in items} == {"user_signal"}
