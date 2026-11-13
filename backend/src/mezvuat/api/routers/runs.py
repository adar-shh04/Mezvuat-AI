"""Research runs (ADR-0008): start a run, read its result, follow its progress over Server-Sent Events.

Events: `run_queued`, `run_started`, `node` (a stage finished), `answer_delta` (streamed answer text),
`answer_reset` (discard the streamed text, a retry follows), `answer_replaced` (the streamed text failed
validation; the final answer replaces it), `answer_final`, then exactly one of `run_completed` or `run_failed`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from mezvuat.api.deps import get_run_service, require_api_key, workspace_data_policy
from mezvuat.api.runs import BusEvent, RunService, sse_format
from mezvuat.api.schemas import RunCreate, RunCreated, RunErrorOut, RunOut
from mezvuat.core.errors import NotFoundError
from mezvuat.inference import DataPolicy
from mezvuat.orchestration import RunFilters

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/runs", tags=["research"], dependencies=[Depends(require_api_key)]
)
RunServiceDep = Annotated[RunService, Depends(get_run_service)]
PolicyDep = Annotated[DataPolicy, Depends(workspace_data_policy)]
KEEPALIVE_S = 15.0


@router.post("", status_code=202, response_model=RunCreated)
def create_run(workspace_id: str, body: RunCreate, service: RunServiceDep, policy: PolicyDep) -> RunCreated:
    filters = RunFilters(
        document_ids=body.filters.document_ids,
        latest_only=body.filters.latest_only,
        published_from=body.filters.published_from,
        published_to=body.filters.published_to,
        include_references=body.filters.include_references,
    )
    run_id = service.start(
        workspace_id=workspace_id, data_policy=policy, question=body.question, mode=body.mode, filters=filters
    )
    base = f"/api/v1/workspaces/{workspace_id}/runs/{run_id}"
    return RunCreated(run_id=run_id, status="queued", run_url=base, events_url=f"{base}/events")


def _load(service: RunService, workspace_id: str, run_id: str) -> RunOut:
    row = service.store.get(workspace_id, run_id)
    if row is None:
        raise NotFoundError("run not found")
    code, message = row.pop("error_code"), row.pop("error_message")
    return RunOut(**row, error=RunErrorOut(code=code, message=message) if code else None)


@router.get("/{run_id}", response_model=RunOut)
def get_run(workspace_id: str, run_id: str, service: RunServiceDep) -> RunOut:
    return _load(service, workspace_id, run_id)


@router.get("/{run_id}/events")
async def run_events(
    workspace_id: str,
    run_id: str,
    request: Request,
    service: RunServiceDep,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    run = await run_in_threadpool(_load, service, workspace_id, run_id)
    after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0

    async def stream() -> AsyncIterator[str]:
        if not service.bus.has(run_id):
            # Not in this process's memory (finished long ago, or the API restarted): report the stored outcome.
            outcome = "run_completed" if run.status == "completed" else "run_failed"
            data = {"run_id": run_id, "status": run.status, "answer": run.answer, "error": run.error}
            if run.error is not None:
                data["error"] = run.error.model_dump()
            yield sse_format(BusEvent(after + 1, outcome, data))
            return
        cursor = after
        while True:
            events, closed = await run_in_threadpool(service.events, run_id, cursor, KEEPALIVE_S)
            for event in events:
                cursor = event.id
                yield sse_format(event)
            if closed and not events:
                return
            if closed:
                continue
            if not events:
                if await request.is_disconnected():
                    return
                yield ": keep-alive\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
