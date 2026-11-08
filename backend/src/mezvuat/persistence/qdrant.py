"""Qdrant adapter (ADR-0005). Every search is scoped to a workspace: the filter is added here, unconditionally."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from mezvuat.core.config import Settings
from mezvuat.core.errors import RetrievalUnavailableError
from mezvuat.retrieval.types import ChunkPayload, Filters, ScoredChunk

_NAMESPACE = uuid.UUID("6f9f2b64-5a5b-4c1a-9d3c-0d3c3f7f1e11")


def _translate_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Connectivity failures become RetrievalUnavailableError, so callers can refuse to answer."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except (ApiException, httpx.TransportError, OSError) as exc:
            raise RetrievalUnavailableError(f"vector store unavailable: {type(exc).__name__}") from exc

    return wrapper


def collection_name(model_id: str, dim: int, index_version: str) -> str:
    slug = re.sub(r"[^a-z0-9.-]+", "-", model_id.split("/")[-1].lower())
    return f"chunks__{slug}__{dim}__{index_version}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _rfc3339(day: Any) -> str:
    return f"{day.isoformat()}T00:00:00Z"


def build_filter(workspace_id: str, filters: Filters) -> models.Filter:
    """The workspace condition is always first and always present. There is no way to omit it."""
    if not workspace_id:
        raise ValueError("workspace_id is required")
    must: list[models.Condition] = [
        models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
    ]
    must_not: list[models.Condition] = []
    if filters.document_ids:
        must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=sorted(filters.document_ids))))
    if filters.latest_only:
        must.append(models.FieldCondition(key="is_latest_version", match=models.MatchValue(value=True)))
    if filters.published_from or filters.published_to:
        must.append(
            models.FieldCondition(
                key="published_at",
                range=models.DatetimeRange(
                    gte=_rfc3339(filters.published_from) if filters.published_from else None,
                    lte=_rfc3339(filters.published_to) if filters.published_to else None,
                ),
            )
        )
    if filters.exclude_section_kinds:
        must_not.append(
            models.FieldCondition(key="section_kind", match=models.MatchAny(any=sorted(filters.exclude_section_kinds)))
        )
    return models.Filter(must=must, must_not=must_not or None)


@dataclass(frozen=True)
class DensePoint:
    workspace_id: str
    payload: ChunkPayload
    vector: list[float]


class QdrantDenseIndex:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self.collection = collection

    @classmethod
    def from_settings(cls, settings: Settings, collection: str) -> QdrantDenseIndex:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=30, check_compatibility=False)
        return cls(client, collection)

    @_translate_errors
    def ping(self) -> bool:
        self._client.get_collections()
        return True

    @_translate_errors
    def ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self.collection):
            return
        try:
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                on_disk_payload=True,
            )
        except UnexpectedResponse as exc:
            # A concurrent worker created it first.
            if exc.status_code != 409:
                raise
            return
        self._client.create_payload_index(
            self.collection,
            "workspace_id",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        for field in ("document_id", "document_version_id", "section_kind"):
            self._client.create_payload_index(self.collection, field, field_schema=models.PayloadSchemaType.KEYWORD)
        self._client.create_payload_index(
            self.collection, "is_latest_version", field_schema=models.PayloadSchemaType.BOOL
        )
        self._client.create_payload_index(
            self.collection, "published_at", field_schema=models.PayloadSchemaType.DATETIME
        )

    @_translate_errors
    def upsert(self, points: Sequence[DensePoint], batch_size: int = 256) -> None:
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self._client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(p.payload.chunk_id),
                        vector=p.vector,
                        payload={
                            "workspace_id": p.workspace_id,
                            "chunk_id": p.payload.chunk_id,
                            "document_id": p.payload.document_id,
                            "document_version_id": p.payload.document_version_id,
                            "is_latest_version": p.payload.is_latest_version,
                            "published_at": p.payload.published_at,
                            "section_kind": p.payload.section_kind,
                        },
                    )
                    for p in batch
                ],
                wait=True,
            )

    @_translate_errors
    def search(self, workspace_id: str, query_vector: list[float], k: int, filters: Filters) -> list[ScoredChunk]:
        response = self._client.query_points(
            self.collection,
            query=query_vector,
            query_filter=build_filter(workspace_id, filters),
            limit=k,
            with_payload=["chunk_id"],
        )
        return [
            ScoredChunk(chunk_id=str((point.payload or {})["chunk_id"]), score=float(point.score))
            for point in response.points
        ]

    @_translate_errors
    def count(self, workspace_id: str) -> int:
        result = self._client.count(
            self.collection,
            count_filter=build_filter(workspace_id, Filters(exclude_section_kinds=frozenset())),
        )
        return int(result.count)

    @_translate_errors
    def delete_workspace(self, workspace_id: str) -> None:
        if not self._client.collection_exists(self.collection):
            return
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=build_filter(workspace_id, Filters(exclude_section_kinds=frozenset()))
            ),
            wait=True,
        )

    @_translate_errors
    def update_latest_flag(self, workspace_id: str, document_version_id: str, is_latest: bool | None) -> None:
        self._client.set_payload(
            self.collection,
            payload={"is_latest_version": is_latest},
            points=models.Filter(
                must=[
                    models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
                    models.FieldCondition(
                        key="document_version_id", match=models.MatchValue(value=document_version_id)
                    ),
                ]
            ),
            wait=True,
        )

    @_translate_errors
    def drop_collection(self) -> None:
        self._client.delete_collection(self.collection)
