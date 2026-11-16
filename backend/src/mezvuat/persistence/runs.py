"""Research-run persistence: the run row, `llm_call` rows, page maps, and the LangGraph checkpointer."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from mezvuat.citations import PageSpan
from mezvuat.core.clock import utcnow
from mezvuat.core.ids import new_id
from mezvuat.orchestration import LLMCallRecord
from mezvuat.persistence.models import DocumentVersion, LlmCall, ResearchRun, RunFeedback, TemporalFact
from mezvuat.temporal import DateFact, VersionDates

log = logging.getLogger(__name__)


class SqlRunStore:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def create(self, *, run_id: str, workspace_id: str, question: str, data_policy: str, retrieval_mode: str) -> None:
        with self._sessions.begin() as session:
            session.add(
                ResearchRun(
                    id=run_id,
                    workspace_id=workspace_id,
                    question=question,
                    status="queued",
                    data_policy=data_policy,
                    retrieval_mode=retrieval_mode,
                )
            )

    def mark_running(self, run_id: str) -> None:
        with self._sessions.begin() as session:
            session.execute(
                update(ResearchRun).where(ResearchRun.id == run_id).values(status="running", started_at=utcnow())
            )

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        answer: dict[str, Any] | None,
        summary: dict[str, Any],
        error_code: str | None,
        error_message: str | None,
    ) -> None:
        with self._sessions.begin() as session:
            session.execute(
                update(ResearchRun)
                .where(ResearchRun.id == run_id)
                .values(
                    status=status,
                    answer=answer,
                    summary=summary,
                    error_code=error_code,
                    error_message=error_message,
                    finished_at=utcnow(),
                )
            )

    def get(self, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            run = session.scalar(
                select(ResearchRun).where(ResearchRun.id == run_id, ResearchRun.workspace_id == workspace_id)
            )
            if run is None:
                return None
            calls = session.scalars(select(LlmCall).where(LlmCall.run_id == run_id).order_by(LlmCall.created_at))
            return {
                "run_id": run.id,
                "workspace_id": run.workspace_id,
                "question": run.question,
                "status": run.status,
                "data_policy": run.data_policy,
                "retrieval_mode": run.retrieval_mode,
                "created_at": run.created_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "error_code": run.error_code,
                "error_message": run.error_message,
                "answer": run.answer,
                "summary": run.summary or {},
                "llm_calls": [
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
                        "modelmesh_request_id": c.modelmesh_request_id,
                        "modelmesh_trace_id": c.modelmesh_trace_id,
                        "fallback_reason": c.fallback_reason,
                        "cache_status": c.cache_status,
                        "cost_estimated_usd": c.cost_estimated_usd,
                    }
                    for c in calls
                ],
            }

    def add_feedback(
        self, workspace_id: str, run_id: str, *, rating: str, claim_id: str | None, comment: str | None
    ) -> dict[str, Any] | None:
        """Store one feedback row. None when the run does not exist in this workspace."""
        with self._sessions.begin() as session:
            exists = session.scalar(
                select(ResearchRun.id).where(ResearchRun.id == run_id, ResearchRun.workspace_id == workspace_id)
            )
            if exists is None:
                return None
            row = RunFeedback(
                id=new_id("fb"),
                run_id=run_id,
                workspace_id=workspace_id,
                rating=rating,
                claim_id=claim_id,
                comment=comment,
                created_at=utcnow(),
            )
            session.add(row)
            return {
                "feedback_id": row.id,
                "run_id": run_id,
                "rating": rating,
                "claim_id": claim_id,
                "created_at": row.created_at,
            }


class SqlCallRecorder:
    """Writes `llm_call` rows as they happen. Never stores prompt or response text."""

    def __init__(self, sessions: sessionmaker[Session], workspace_id: str) -> None:
        self._sessions = sessions
        self._workspace_id = workspace_id

    def record(self, records: Sequence[LLMCallRecord]) -> None:
        with self._sessions.begin() as session:
            for r in records:
                session.add(
                    LlmCall(
                        id=r.client_request_id,
                        workspace_id=self._workspace_id,
                        run_id=r.run_id,
                        workflow_node=r.workflow_node,
                        task_type=r.task_type,
                        backend=r.backend,
                        model=r.model,
                        client_request_id=r.client_request_id,
                        modelmesh_request_id=r.modelmesh_request_id,
                        modelmesh_trace_id=r.modelmesh_trace_id,
                        fallback_reason=r.fallback_reason,
                        langfuse_trace_id=r.langfuse_trace_id,
                        prompt_tokens=r.prompt_tokens,
                        completion_tokens=r.completion_tokens,
                        usage_source=r.usage_source,
                        cost_estimated_usd=r.cost_estimated_usd,
                        latency_ms=r.latency_ms,
                        cache_status=r.cache_status,
                        provider=r.provider,
                        prompt_version=r.prompt_version,
                        iteration=r.iteration,
                        status=r.status,
                        error_code=r.error_code,
                        http_status=r.http_status,
                        finish_reason=r.finish_reason,
                        structured_mode=r.structured_mode,
                    )
                )


class PostgresPageMap:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def page_spans(self, workspace_id: str, version_ids: Sequence[str]) -> dict[str, list[PageSpan]]:
        if not version_ids:
            return {}
        with self._sessions() as session:
            rows = session.execute(
                select(DocumentVersion.id, DocumentVersion.pages).where(
                    DocumentVersion.workspace_id == workspace_id, DocumentVersion.id.in_(list(version_ids))
                )
            ).all()
        return {
            version_id: [PageSpan(page=p["page"], start=p["start"], end=p["end"]) for p in pages]
            for version_id, pages in rows
            if pages
        }


class PostgresTemporalSource:
    """All non-failed versions of the given documents with their typed dates. `ingested_at` is never included."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def versions(self, workspace_id: str, document_ids: Sequence[str]) -> dict[str, list[VersionDates]]:
        if not document_ids:
            return {}
        with self._sessions() as session:
            versions = session.execute(
                select(
                    DocumentVersion.id,
                    DocumentVersion.document_id,
                    DocumentVersion.version_label,
                    DocumentVersion.version_ordinal,
                    DocumentVersion.is_latest_version,
                ).where(
                    DocumentVersion.workspace_id == workspace_id,
                    DocumentVersion.document_id.in_(list(document_ids)),
                    DocumentVersion.ingestion_status != "failed",
                )
            ).all()
            facts = session.scalars(
                select(TemporalFact).where(TemporalFact.document_version_id.in_([v.id for v in versions]))
            ).all()
        by_version: dict[str, list[DateFact]] = {}
        for fact in facts:
            by_version.setdefault(fact.document_version_id, []).append(
                DateFact.model_validate(
                    {
                        "kind": fact.kind,
                        "value": fact.value,
                        "precision": fact.precision,
                        "source": fact.source,
                        "confidence": fact.confidence,
                    }
                )
            )
        result: dict[str, list[VersionDates]] = {}
        for version_id, document_id, label, ordinal, latest in versions:
            result.setdefault(document_id, []).append(
                VersionDates(
                    version_id=version_id,
                    version_label=label,
                    version_ordinal=ordinal,
                    is_latest_version=latest,
                    facts=by_version.get(version_id, []),
                )
            )
        return result


def postgres_checkpointer(dsn: str, serde: SerializerProtocol, max_size: int = 4) -> PostgresSaver:
    """A pooled sync checkpointer. `setup()` creates LangGraph's own tables (it versions them itself)."""
    pool: ConnectionPool[Any] = ConnectionPool(
        dsn,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool, serde=serde)
    saver.setup()
    return saver
