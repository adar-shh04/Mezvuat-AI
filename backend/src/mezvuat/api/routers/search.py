from __future__ import annotations

from fastapi import APIRouter, Depends

from mezvuat.api.deps import (
    SessionDep,
    get_dense_index,
    get_embedder,
    get_lexical_store,
    get_reranker,
    require_api_key,
)
from mezvuat.api.schemas import EvidenceOut, SearchRequest, SearchResponse
from mezvuat.persistence.repositories import PostgresChunkStore
from mezvuat.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from mezvuat.retrieval.types import Filters, PassageReranker
from mezvuat.workspaces.service import get_workspace

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}", tags=["retrieval"], dependencies=[Depends(require_api_key)]
)


class _LazyEmbedder:
    """Loads the embedding model on first use, so lexical-only searches never pay for it."""

    def embed_query(self, query: str) -> list[float]:
        return get_embedder().embed_query(query)


@router.post("/search", response_model=SearchResponse)
def search(workspace_id: str, body: SearchRequest, session: SessionDep) -> SearchResponse:
    get_workspace(session, workspace_id)
    reranker: PassageReranker | None = get_reranker() if body.mode == "hybrid_rrf_rerank" else None
    pipeline = RetrievalPipeline(
        lexical=get_lexical_store(),
        dense=get_dense_index(),
        embedder=_LazyEmbedder(),
        store=PostgresChunkStore(session),
        reranker=reranker,
    )
    filters = Filters(
        document_ids=frozenset(body.filters.document_ids),
        latest_only=body.filters.latest_only,
        published_from=body.filters.published_from,
        published_to=body.filters.published_to,
        exclude_section_kinds=frozenset() if body.filters.include_references else frozenset({"references"}),
    )
    result = pipeline.retrieve(
        workspace_id,
        body.query,
        k=body.k,
        config=RetrievalConfig(mode=body.mode, candidate_depth=body.candidate_depth, rerank_depth=body.rerank_depth),
        filters=filters,
    )
    return SearchResponse(
        mode=result.mode,
        timings_ms=result.timings_ms,
        evidence=[
            EvidenceOut(
                rank=e.rank,
                score=e.score,
                chunk_id=e.chunk.chunk_id,
                document_id=e.chunk.document_id,
                document_version_id=e.chunk.document_version_id,
                document_title=e.chunk.document_title,
                version_label=e.chunk.version_label,
                is_latest_version=e.chunk.is_latest_version,
                text=e.chunk.text,
                char_start=e.chunk.char_start,
                char_end=e.chunk.char_end,
                page_start=e.chunk.page_start,
                page_end=e.chunk.page_end,
                section_path=list(e.chunk.section_path),
                section_kind=e.chunk.section_kind,
                lexical_rank=e.lexical_rank,
                dense_rank=e.dense_rank,
                fusion_score=e.fusion_score,
                rerank_score=e.rerank_score,
            )
            for e in result.evidence
        ],
    )
