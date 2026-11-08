"""Workspace lifecycle. Deleting a workspace removes its rows, vectors, lexical index and unshared blobs."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from mezvuat.core.errors import NotFoundError
from mezvuat.core.ids import new_id
from mezvuat.persistence.blobstore import BlobStore
from mezvuat.persistence.models import DocumentVersion, Workspace
from mezvuat.persistence.qdrant import QdrantDenseIndex
from mezvuat.retrieval.lexical import WorkspaceLexicalIndexStore


def create_workspace(session: Session, name: str) -> Workspace:
    workspace = Workspace(id=new_id("ws"), name=name, data_classification="private")
    session.add(workspace)
    session.commit()
    return workspace


def get_workspace(session: Session, workspace_id: str) -> Workspace:
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise NotFoundError("workspace not found")
    return workspace


def list_workspaces(session: Session) -> Sequence[Workspace]:
    return session.scalars(select(Workspace).order_by(Workspace.created_at.desc())).all()


def delete_workspace(
    session: Session,
    workspace_id: str,
    *,
    dense: QdrantDenseIndex,
    lexical: WorkspaceLexicalIndexStore,
    blobs: BlobStore,
) -> None:
    workspace = get_workspace(session, workspace_id)
    digests = {
        digest
        for digest in session.scalars(
            select(DocumentVersion.raw_sha256).where(DocumentVersion.workspace_id == workspace_id)
        )
        if digest
    }
    # Vectors go first: if this fails, the rows remain and the delete can simply be retried.
    dense.delete_workspace(workspace_id)
    session.delete(workspace)
    session.commit()
    lexical.delete_workspace(workspace_id)
    for digest in digests:
        still_used = session.scalar(select(DocumentVersion.id).where(DocumentVersion.raw_sha256 == digest).limit(1))
        if still_used is None:
            blobs.delete(digest)
