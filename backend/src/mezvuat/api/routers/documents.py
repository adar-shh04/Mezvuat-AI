from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload, undefer

from mezvuat.api.deps import SessionDep, SettingsDep, get_blob_store, require_api_key
from mezvuat.api.schemas import DocumentOut, DocumentVersionOut, SubmitOut, UrlIngestRequest, VersionTextOut
from mezvuat.api.text_view import select_window
from mezvuat.core.errors import NotFoundError, PayloadTooLargeError
from mezvuat.ingestion import service
from mezvuat.ingestion.service import SubmitResult
from mezvuat.persistence.models import Document, DocumentVersion
from mezvuat.workspaces.service import get_workspace

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)


def _submit_response(result: SubmitResult, session: SessionDep) -> JSONResponse:
    loaded = (
        session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.id == result.version.id)
            .options(joinedload(DocumentVersion.temporal_facts))
            .execution_options(populate_existing=True)
        )
        .unique()
        .one()
    )
    body = SubmitOut(
        status="duplicate" if result.duplicate else "accepted",
        version=DocumentVersionOut.model_validate(loaded),
    )
    code = status.HTTP_200_OK if result.duplicate else status.HTTP_202_ACCEPTED
    return JSONResponse(status_code=code, content=body.model_dump(mode="json"))


@router.post("", response_model=SubmitOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    workspace_id: str,
    file: Annotated[UploadFile, File()],
    session: SessionDep,
    settings: SettingsDep,
) -> JSONResponse:
    raw = file.file.read(settings.upload_max_bytes + 1)
    if len(raw) > settings.upload_max_bytes:
        raise PayloadTooLargeError(f"file exceeds the {settings.upload_max_bytes} byte limit")
    result = service.submit_file(session, get_blob_store(), settings, workspace_id, raw, file.filename)
    return _submit_response(result, session)


@router.post("/url", response_model=SubmitOut, status_code=status.HTTP_202_ACCEPTED)
def ingest_url(workspace_id: str, body: UrlIngestRequest, session: SessionDep) -> JSONResponse:
    result = service.submit_url(session, workspace_id, str(body.url))
    return _submit_response(result, session)


@router.get("", response_model=list[DocumentOut])
def list_documents(workspace_id: str, session: SessionDep) -> Sequence[Document]:
    get_workspace(session, workspace_id)
    return (
        session.scalars(
            select(Document)
            .where(Document.workspace_id == workspace_id)
            .options(selectinload(Document.versions).selectinload(DocumentVersion.temporal_facts))
            .order_by(Document.created_at)
        )
        .unique()
        .all()
    )


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(workspace_id: str, document_id: str, session: SessionDep) -> Document:
    document = (
        session.scalars(
            select(Document)
            .where(Document.workspace_id == workspace_id, Document.id == document_id)
            .options(selectinload(Document.versions).selectinload(DocumentVersion.temporal_facts))
        )
        .unique()
        .one_or_none()
    )
    if document is None:
        raise NotFoundError("document not found")
    return document


@router.get("/{document_id}/versions/{version_id}/text", response_model=VersionTextOut)
def get_version_text(
    workspace_id: str,
    document_id: str,
    version_id: str,
    session: SessionDep,
    page: Annotated[int | None, Query(ge=1)] = None,
    at: Annotated[
        int | None, Query(ge=0, description="An absolute character offset; returns the page holding it")
    ] = None,
) -> VersionTextOut:
    """One page of the normalized text, for the evidence viewer. Citation offsets index into the same text."""
    version = session.scalar(
        select(DocumentVersion)
        .where(
            DocumentVersion.workspace_id == workspace_id,
            DocumentVersion.document_id == document_id,
            DocumentVersion.id == version_id,
        )
        .options(undefer(DocumentVersion.normalized_text), undefer(DocumentVersion.pages))
    )
    if version is None:
        raise NotFoundError("document version not found")
    if version.normalized_text is None:
        raise NotFoundError("this version has no extracted text yet")
    text = version.normalized_text
    window = select_window(len(text), version.pages, page=page, at=at)
    return VersionTextOut(
        document_id=document_id,
        document_version_id=version_id,
        page=window.page,
        page_count=window.page_count,
        start=window.start,
        end=window.end,
        text=text[window.start : window.end],
    )
