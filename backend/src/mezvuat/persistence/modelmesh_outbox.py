"""The ModelMesh outbox (M8): side effects that must happen even if ModelMesh is down when they are decided.

- Workspace deletion queues `POST /v1/cache/invalidate {namespace}` and `DELETE /v1/feedback/{mmr id}` for every
  ModelMesh request the workspace made, in the same transaction as the delete. The `llm_call` rows lose their
  workspace id on delete (`SET NULL`), so the ids are collected first.
- User ratings queue a `POST /v1/feedback/batch` of `user_thumbs` labels (`user_signal`, `attribution: workflow`).
The worker drains the queue in creation order, stops at the first sign that ModelMesh is unavailable, and marks a row
failed after `max_attempts`. Rows hold ids and scores only, never prompt or document text.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mezvuat.core.clock import utcnow
from mezvuat.core.errors import MezvuatError
from mezvuat.core.ids import new_id
from mezvuat.inference import InferenceUnavailableError
from mezvuat.inference.modelmesh import FeedbackItem, FeedbackOutcome, user_thumbs_items
from mezvuat.persistence.models import LlmCall, ModelMeshOutbox

log = logging.getLogger(__name__)


class OutboxOps(Protocol):
    def invalidate_cache(self, *, namespace: str | None = None, context_fingerprint: str | None = None) -> int: ...

    def delete_feedback(self, request_id: str) -> int: ...

    def post_feedback(self, items: Sequence[FeedbackItem]) -> FeedbackOutcome: ...


def _add(session: Session, kind: str, payload: dict[str, Any]) -> None:
    session.add(ModelMeshOutbox(id=new_id("mmo"), kind=kind, payload=payload, status="queued", attempts=0))


def workspace_request_ids(session: Session, workspace_id: str) -> list[str]:
    rows = session.scalars(
        select(LlmCall.modelmesh_request_id)
        .where(LlmCall.workspace_id == workspace_id, LlmCall.modelmesh_request_id.is_not(None))
        .distinct()
    )
    return sorted(r for r in rows if r)


def enqueue_workspace_cleanup(session: Session, workspace_id: str, namespace: str | None) -> int:
    """Queue the ModelMesh clean-up for a workspace about to be deleted. Call inside the deleting transaction."""
    request_ids = workspace_request_ids(session, workspace_id)
    if namespace is not None:
        _add(session, "cache_invalidate", {"namespace": namespace})
    for request_id in request_ids:
        _add(session, "feedback_delete", {"request_id": request_id})
    return len(request_ids) + (namespace is not None)


def enqueue_run_thumbs(
    sessions: sessionmaker[Session], run_id: str, rating: Literal["up", "down"], labelled_at: datetime
) -> int:
    """Queue a thumbs rating for every ModelMesh request of a run. Returns the number of labels queued."""
    with sessions.begin() as session:
        request_ids = session.scalars(
            select(LlmCall.modelmesh_request_id).where(
                LlmCall.run_id == run_id, LlmCall.modelmesh_request_id.is_not(None), LlmCall.status == "ok"
            )
        ).all()
        items = user_thumbs_items([r for r in request_ids if r], rating=rating, labelled_at=labelled_at)
        if items:
            _add(session, "feedback_post", {"items": [item.wire() for item in items]})
        return len(items)


def enqueue_feedback(sessions: sessionmaker[Session], items: Sequence[FeedbackItem], batch: int = 500) -> int:
    """Queue evaluation labels (M6) for `POST /v1/feedback/batch`, at most `batch` per outbox row."""
    with sessions.begin() as session:
        for start in range(0, len(items), batch):
            _add(session, "feedback_post", {"items": [item.wire() for item in items[start : start + batch]]})
    return len(items)


def _run(ops: OutboxOps, row: ModelMeshOutbox) -> None:
    payload = row.payload
    if row.kind == "cache_invalidate":
        ops.invalidate_cache(namespace=payload["namespace"])
    elif row.kind == "feedback_delete":
        ops.delete_feedback(payload["request_id"])
    elif row.kind == "feedback_post":
        outcome = ops.post_feedback([FeedbackItem.model_validate(item) for item in payload["items"]])
        if outcome.conflicts:
            log.error(
                "feedback conflicts were not overwritten", extra={"outbox_id": row.id, "count": len(outcome.conflicts)}
            )
    else:
        raise ValueError(f"unknown outbox kind {row.kind}")


def drain_outbox(sessions: sessionmaker[Session], ops: OutboxOps, *, limit: int = 20, max_attempts: int = 8) -> int:
    """Process up to `limit` queued rows. Returns how many finished (done or failed)."""
    finished = 0
    with sessions() as session:
        rows = session.scalars(
            select(ModelMeshOutbox)
            .where(ModelMeshOutbox.status == "queued")
            .order_by(ModelMeshOutbox.created_at, ModelMeshOutbox.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
        for row in rows:
            row.attempts += 1
            try:
                _run(ops, row)
            except InferenceUnavailableError as exc:
                # ModelMesh is down: keep the order and try again on the next drain.
                row.error_code = exc.code
                if row.attempts >= max_attempts:
                    row.status, row.finished_at = "failed", utcnow()
                    finished += 1
                session.commit()
                log.warning("ModelMesh outbox paused: ModelMesh unavailable", extra={"outbox_id": row.id})
                break
            except (MezvuatError, ValueError, KeyError) as exc:
                row.error_code = getattr(exc, "code", type(exc).__name__)[:64]
                if row.attempts >= max_attempts:
                    row.status, row.finished_at = "failed", utcnow()
                    finished += 1
                log.warning("ModelMesh outbox item failed", extra={"outbox_id": row.id, "kind": row.kind})
            else:
                row.status, row.error_code, row.finished_at = "done", None, utcnow()
                finished += 1
            session.commit()
    return finished
