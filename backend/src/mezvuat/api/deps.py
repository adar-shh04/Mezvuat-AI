"""Process-wide services and FastAPI dependencies. ML models load lazily on first search."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from mezvuat.core.config import Settings, get_settings
from mezvuat.core.errors import MezvuatError
from mezvuat.ml.models import CrossEncoderReranker, SentenceTransformerEmbedder
from mezvuat.ml.registry import get_embedder_spec
from mezvuat.persistence.blobstore import LocalBlobStore
from mezvuat.persistence.db import get_session_factory
from mezvuat.persistence.models import Workspace
from mezvuat.persistence.qdrant import QdrantDenseIndex, collection_name
from mezvuat.retrieval.lexical import WorkspaceLexicalIndexStore


class UnauthorizedError(MezvuatError):
    code = "unauthorized"
    http_status = 401


def get_session() -> Iterator[Session]:
    with get_session_factory()() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def require_api_key(settings: SettingsDep, authorization: Annotated[str | None, Header()] = None) -> None:
    """Optional single shared key. With MEZVUAT_API_KEY unset the API is open; compose binds it to 127.0.0.1."""
    expected = settings.mezvuat_api_key
    if expected is None or not expected.get_secret_value():
        return
    presented = (authorization or "").removeprefix("Bearer ")
    if not secrets.compare_digest(presented.encode(), expected.get_secret_value().encode()):
        raise UnauthorizedError("missing or invalid API key")


@lru_cache
def get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(get_settings().blob_dir)


@lru_cache
def get_dense_index() -> QdrantDenseIndex:
    settings = get_settings()
    spec = get_embedder_spec(settings.embedding_model_id)
    return QdrantDenseIndex.from_settings(settings, collection_name(spec.model_id, spec.dim, settings.index_version))


def _current_lexical_version(workspace_id: str) -> int | None:
    with get_session_factory()() as session:
        workspace = session.get(Workspace, workspace_id)
        return workspace.lexical_index_version if workspace else None


@lru_cache
def get_lexical_store() -> WorkspaceLexicalIndexStore:
    return WorkspaceLexicalIndexStore(get_settings().lexical_dir, _current_lexical_version)


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    settings = get_settings()
    return SentenceTransformerEmbedder.from_model_id(settings.embedding_model_id, device=settings.ml_device)


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker.from_model_id(settings.reranker_model_id, device=settings.ml_device)
