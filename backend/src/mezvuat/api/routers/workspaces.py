from __future__ import annotations

from collections.abc import Sequence

from fastapi import APIRouter, Depends, Response, status

from mezvuat.api.deps import SessionDep, get_blob_store, get_dense_index, get_lexical_store, require_api_key
from mezvuat.api.schemas import WorkspaceCreate, WorkspaceOut
from mezvuat.persistence.models import Workspace
from mezvuat.workspaces import service

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED)
def create_workspace(body: WorkspaceCreate, session: SessionDep) -> Workspace:
    return service.create_workspace(session, body.name)


@router.get("", response_model=list[WorkspaceOut])
def list_workspaces(session: SessionDep) -> Sequence[Workspace]:
    return service.list_workspaces(session)


@router.get("/{workspace_id}", response_model=WorkspaceOut)
def get_workspace(workspace_id: str, session: SessionDep) -> Workspace:
    return service.get_workspace(session, workspace_id)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(workspace_id: str, session: SessionDep) -> Response:
    service.delete_workspace(
        session,
        workspace_id,
        dense=get_dense_index(),
        lexical=get_lexical_store(),
        blobs=get_blob_store(),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
