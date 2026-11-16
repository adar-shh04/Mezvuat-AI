"""Process-wide services and FastAPI dependencies. ML models load lazily on first search."""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, Header
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.orm import Session

from mezvuat.api.runs import RunLimits, RunService
from mezvuat.citations import NliScores, load_thresholds
from mezvuat.core.config import Settings, get_settings
from mezvuat.core.errors import MezvuatError, NotFoundError
from mezvuat.inference import DataPolicy, InferenceRejectedError, LLMClient, build_llm_client, namespace_secret
from mezvuat.inference.modelmesh import ModelMeshClient, cache_namespace
from mezvuat.ml.models import CrossEncoderReranker, SentenceTransformerEmbedder
from mezvuat.ml.registry import get_embedder_spec
from mezvuat.orchestration import RunFilters, WorkflowDeps, checkpoint_serializer, memory_checkpointer
from mezvuat.persistence.blobstore import LocalBlobStore
from mezvuat.persistence.db import get_session_factory
from mezvuat.persistence.models import Workspace
from mezvuat.persistence.qdrant import QdrantDenseIndex, collection_name
from mezvuat.persistence.repositories import PostgresChunkStore
from mezvuat.persistence.runs import (
    PostgresPageMap,
    PostgresTemporalSource,
    SqlCallRecorder,
    SqlRunStore,
    postgres_checkpointer,
)
from mezvuat.retrieval.lexical import WorkspaceLexicalIndexStore
from mezvuat.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from mezvuat.retrieval.types import Filters, RankedEvidence, RetrievalMode


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


_nli_lock = threading.Lock()
_nli: Any = None


def get_nli() -> Any:
    """The NLI model, loaded once. Loading takes tens of seconds on CPU, so the API preloads it at start-up."""
    global _nli
    with _nli_lock:
        if _nli is None:
            from mezvuat.ml.nli import CrossEncoderNli

            settings = get_settings()
            _nli = CrossEncoderNli.from_model_id(settings.nli_model_id, device=settings.nli_device)
        return _nli


def preload_nli() -> None:
    try:
        get_nli()
    except Exception:  # The verifier falls back to the quote check; a failed preload must not stop the API.
        logging.getLogger("mezvuat.api").exception("NLI model preload failed; claims will keep the quote check")


class LazyNli:
    """Loads the NLI model on the first claim to verify. A load failure surfaces in `verify`, which falls back."""

    @property
    def model_id(self) -> str:
        return get_settings().nli_model_id

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliScores]:
        scores: list[NliScores] = get_nli().predict(pairs)
        return scores


class LazyEmbedder:
    """Loads the embedding model on first use, so lexical-only searches never pay for it."""

    def embed_query(self, query: str) -> list[float]:
        return get_embedder().embed_query(query)


def to_filters(filters: RunFilters) -> Filters:
    return Filters(
        document_ids=frozenset(filters.document_ids),
        latest_only=filters.latest_only,
        published_from=filters.published_from,
        published_to=filters.published_to,
        exclude_section_kinds=frozenset() if filters.include_references else frozenset({"references"}),
    )


class PipelineSearch:
    """The production retrieval pipeline behind the graph's `EvidenceSearch` protocol."""

    def __init__(self, pipeline: RetrievalPipeline, config: RetrievalConfig) -> None:
        self._pipeline = pipeline
        self._config = config

    def search(self, workspace_id: str, query: str, k: int, filters: RunFilters) -> list[RankedEvidence]:
        return self._pipeline.retrieve(
            workspace_id, query, k=k, config=self._config, filters=to_filters(filters)
        ).evidence


@lru_cache
def get_llm_client() -> LLMClient:
    return build_llm_client(get_settings())


def negotiate_modelmesh() -> None:
    """Start-up negotiation (ADR-0011). A contract mismatch is a hard failure: the API does not start."""
    client = get_llm_client()
    if isinstance(client, ModelMeshClient):
        client.negotiate()


def modelmesh_namespace(settings: Settings, workspace_id: str) -> str | None:
    """The workspace's ModelMesh cache namespace, or None when ModelMesh or the secret is not configured."""
    if settings.modelmesh_url is None:
        return None
    try:
        return cache_namespace(namespace_secret(settings), workspace_id)
    except InferenceRejectedError:
        return None


_checkpointer_lock = threading.Lock()
_checkpointer: BaseCheckpointSaver[Any] | None = None


def get_checkpointer() -> BaseCheckpointSaver[Any]:
    """Created on first use; `RUN_CHECKPOINTER=memory` keeps checkpoints in process (tests, quick local runs)."""
    global _checkpointer
    with _checkpointer_lock:
        if _checkpointer is None:
            settings = get_settings()
            if settings.run_checkpointer == "memory":
                _checkpointer = memory_checkpointer()
            else:
                _checkpointer = postgres_checkpointer(settings.psycopg_dsn, checkpoint_serializer())
        return _checkpointer


@contextmanager
def production_workflow(run_id: str, workspace_id: str, mode: RetrievalMode) -> Iterator[WorkflowDeps]:
    settings = get_settings()
    sessions = get_session_factory()
    with sessions() as session:
        pipeline = RetrievalPipeline(
            lexical=get_lexical_store(),
            dense=get_dense_index(),
            embedder=LazyEmbedder(),
            store=PostgresChunkStore(session),
            reranker=get_reranker() if mode == "hybrid_rrf_rerank" else None,
        )
        yield WorkflowDeps(
            llm=get_llm_client(),
            search=PipelineSearch(pipeline, RetrievalConfig(mode=mode)),
            pages=PostgresPageMap(sessions),
            temporal=PostgresTemporalSource(sessions),
            recorder=SqlCallRecorder(sessions, workspace_id),
            max_call_timeout_s=min(60.0, settings.run_deadline_s),
            nli=LazyNli() if settings.nli_enabled else None,
            nli_thresholds=load_thresholds(settings.nli_thresholds_file),
            llm_judge=settings.verify_llm_judge,
            max_judge_calls=settings.verify_max_judge_calls,
        )


@lru_cache
def get_run_service() -> RunService:
    settings = get_settings()
    return RunService(
        store=SqlRunStore(get_session_factory()),
        workflow=production_workflow,
        checkpointer=get_checkpointer,
        limits=RunLimits(
            deadline_s=settings.run_deadline_s,
            max_llm_calls=settings.run_max_llm_calls,
            max_tokens=settings.run_max_tokens,
        ),
        concurrency=settings.run_concurrency,
    )


def workspace_data_policy(workspace_id: str, session: SessionDep) -> DataPolicy:
    """Decision D1: derived from the verified classification, never from the request."""
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise NotFoundError("workspace not found")
    return "any" if workspace.data_classification == "public" else "local_only"
