"""Worker-side ingestion: parse, chunk, embed, index, record."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np
import numpy.typing as npt
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from mezvuat.core.config import Settings
from mezvuat.core.errors import ParseFailedError, ValidationFailedError
from mezvuat.ingestion.chunker import (
    CHUNKER_VERSION,
    TokenCounter,
    chunk_document,
    chunk_id_for,
    embedding_text,
)
from mezvuat.ingestion.parsers.base import ParsedDocument, Parser
from mezvuat.ingestion.parsers.html import HtmlParser
from mezvuat.ingestion.parsers.pdf import PdfParser
from mezvuat.ingestion.parsers.text import MarkdownParser, PlainTextParser
from mezvuat.ingestion.ssrf import Resolver, fetch_url, system_resolver
from mezvuat.ingestion.validation import validate_pdf
from mezvuat.persistence.blobstore import BlobStore
from mezvuat.persistence.models import Chunk, Document, DocumentVersion, Workspace
from mezvuat.persistence.qdrant import DensePoint, QdrantDenseIndex
from mezvuat.persistence.repositories import (
    chunks_for_lexical_index,
    enqueue_lexical_rebuild,
    find_text_duplicate,
    find_version_by_raw,
    mark_versions_ready,
    published_dates,
    recompute_classification,
    recompute_latest_versions,
    recompute_supersedes,
)
from mezvuat.retrieval.lexical import WorkspaceLexicalIndexStore
from mezvuat.retrieval.types import ChunkPayload

log = logging.getLogger(__name__)

_PARSERS: dict[str, Parser] = {
    "pdf": PdfParser(),
    "markdown": MarkdownParser(),
    "text": PlainTextParser(),
    "html": HtmlParser(),
}
_URL_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/plain": "text",
}


class DocumentEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]: ...


@dataclass
class IngestionContext:
    settings: Settings
    blobs: BlobStore
    dense: QdrantDenseIndex
    lexical: WorkspaceLexicalIndexStore
    embedder: DocumentEmbedder
    token_counter: TokenCounter
    http_client: httpx.Client
    resolver: Resolver = system_resolver


def _fetch_url_version(session: Session, ctx: IngestionContext, version: DocumentVersion) -> bool:
    """Fetch and store a URL. Returns False when the content duplicates an existing version."""
    if version.source_uri is None:
        raise ValidationFailedError("URL version has no source", code="missing_source")
    fetched = fetch_url(
        version.source_uri,
        client=ctx.http_client,
        timeout_s=ctx.settings.url_fetch_timeout_s,
        max_bytes=ctx.settings.url_fetch_max_bytes,
        resolver=ctx.resolver,
    )
    digest = ctx.blobs.put(fetched.body)
    twin = find_version_by_raw(session, version.workspace_id, digest)
    if twin is not None and twin.id != version.id:
        version.ingestion_status = "failed"
        version.error_code = "duplicate"
        version.error_message = f"same content as version {twin.id}"
        return False
    version.raw_sha256 = digest
    version.size_bytes = len(fetched.body)
    version.media_type = fetched.content_type
    version.source_uri = fetched.final_url
    version.source_type = _URL_CONTENT_TYPES[fetched.content_type]
    if version.source_type == "pdf":
        validate_pdf(fetched.body, ctx.settings.pdf_max_pages)
    return True


def parse_version(ctx: IngestionContext, version: DocumentVersion) -> ParsedDocument:
    if version.raw_sha256 is None:
        raise ValidationFailedError("version has no stored content", code="missing_source")
    raw = ctx.blobs.get(version.raw_sha256)
    parsed = _PARSERS[version.source_type].parse(raw, filename=version.original_filename)
    if not parsed.text.strip():
        raise ParseFailedError("document contains no extractable text", code="no_text_layer")
    return parsed


def ingest_version(session: Session, ctx: IngestionContext, version_id: str) -> None:
    version = session.get(DocumentVersion, version_id)
    if version is None:
        raise ValidationFailedError("version no longer exists", code="missing_source")
    version.ingestion_status = "processing"
    version.error_code = version.error_message = None
    session.commit()

    if version.source_type == "url" and not _fetch_url_version(session, ctx, version):
        recompute_classification(session, version.workspace_id)
        session.commit()
        return
    session.commit()

    parsed = parse_version(ctx, version)
    drafts = chunk_document(parsed, ctx.token_counter)
    if not drafts:
        raise ParseFailedError("document produced no chunks", code="no_chunks")

    document = session.get_one(Document, version.document_id)
    if document.title in (None, version.original_filename, version.source_uri) and parsed.title:
        document.title = parsed.title
    text_digest = hashlib.sha256(parsed.text.encode()).hexdigest()
    twin = find_text_duplicate(session, version.workspace_id, text_digest, version.id)

    published = published_dates(session, [version.id]).get(version.id)
    published_at = f"{published.isoformat()}T00:00:00Z" if published else None
    ids = [chunk_id_for(version.id, CHUNKER_VERSION, d.ordinal) for d in drafts]
    versions = recompute_latest_versions(session, version.document_id)
    is_latest = next((v.is_latest_version for v in versions if v.id == version.id), None)

    vectors = ctx.embedder.embed_documents([embedding_text(document.title, d.section_path, d.text) for d in drafts])
    ctx.dense.ensure_collection(ctx.embedder.dim)
    ctx.dense.upsert(
        [
            DensePoint(
                workspace_id=version.workspace_id,
                payload=ChunkPayload(
                    chunk_id=chunk_id,
                    document_id=version.document_id,
                    document_version_id=version.id,
                    is_latest_version=is_latest,
                    published_at=published_at,
                    section_kind=draft.section_kind,
                ),
                vector=[float(x) for x in vector],
            )
            for chunk_id, draft, vector in zip(ids, drafts, vectors, strict=True)
        ]
    )
    for other in versions:
        if other.id != version.id:
            ctx.dense.update_latest_flag(version.workspace_id, other.id, other.is_latest_version)

    session.execute(delete(Chunk).where(Chunk.document_version_id == version.id))
    session.add_all(
        Chunk(
            id=chunk_id,
            workspace_id=version.workspace_id,
            document_id=version.document_id,
            document_version_id=version.id,
            ordinal=draft.ordinal,
            char_start=draft.char_start,
            char_end=draft.char_end,
            page_start=draft.page_start,
            page_end=draft.page_end,
            section_path=list(draft.section_path),
            section_kind=draft.section_kind,
            token_count=draft.token_count,
            text=draft.text,
            text_sha256=draft.text_sha256,
            chunker_version=CHUNKER_VERSION,
        )
        for chunk_id, draft in zip(ids, drafts, strict=True)
    )
    version.normalized_text = parsed.text
    version.pages = [{"page": p.page_number, "start": p.char_start, "end": p.char_end} for p in parsed.pages]
    version.sections = [
        {"path": list(s.path), "start": s.char_start, "end": s.char_end, "kind": s.kind} for s in parsed.sections
    ]
    version.text_sha256 = text_digest
    version.probable_duplicate_of = twin.id if twin is not None else None
    version.extractor = parsed.extractor
    version.chunker_version = CHUNKER_VERSION
    version.ingestion_status = "indexed_dense"
    recompute_supersedes(session, version.workspace_id, version.document_id)
    enqueue_lexical_rebuild(session, version.workspace_id)
    recompute_classification(session, version.workspace_id)
    session.commit()
    log.info("version indexed", extra={"version_id": version.id, "chunks": len(drafts)})


def rebuild_lexical_index(session: Session, ctx: IngestionContext, workspace_id: str) -> None:
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        return
    rows, version_ids = chunks_for_lexical_index(session, workspace_id)
    if not rows:
        workspace.lexical_index_version = None
        session.commit()
        return
    next_version = (workspace.lexical_index_version or 0) + 1
    ctx.lexical.write(workspace_id, next_version, [r.payload for r in rows], [r.text for r in rows])
    workspace.lexical_index_version = next_version
    mark_versions_ready(session, version_ids)
    session.commit()
    ctx.lexical.prune(workspace_id, next_version)
    log.info("lexical index rebuilt", extra={"workspace_id": workspace_id, "chunks": len(rows)})


def mark_version_failed(session: Session, version_id: str, code: str, message: str) -> None:
    version = session.get(DocumentVersion, version_id)
    if version is None:
        return
    version.ingestion_status = "failed"
    version.error_code = code
    version.error_message = message[:2000]
    session.flush()
    recompute_latest_versions(session, version.document_id)
    recompute_classification(session, version.workspace_id)


def workspace_blob_digests(session: Session, workspace_id: str) -> list[str]:
    return [
        digest
        for digest in session.scalars(
            select(DocumentVersion.raw_sha256).where(DocumentVersion.workspace_id == workspace_id)
        )
        if digest
    ]
