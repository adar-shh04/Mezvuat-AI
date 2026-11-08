"""Query helpers. Everything that touches SQL beyond a primary-key lookup lives here."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, joinedload

from mezvuat.core.clock import utcnow
from mezvuat.core.ids import new_id
from mezvuat.persistence.models import (
    Chunk,
    Document,
    DocumentRelation,
    DocumentVersion,
    IngestionJob,
    TemporalFact,
    Workspace,
)
from mezvuat.retrieval.types import ChunkPayload, ChunkRecord

READY_STATUSES = ("indexed_dense", "ready")


def find_version_by_raw(session: Session, workspace_id: str, raw_sha256: str) -> DocumentVersion | None:
    return session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.workspace_id == workspace_id, DocumentVersion.raw_sha256 == raw_sha256
        )
    )


def find_document_by_key(session: Session, workspace_id: str, canonical_key: str) -> Document | None:
    return session.scalar(
        select(Document).where(Document.workspace_id == workspace_id, Document.canonical_key == canonical_key)
    )


def find_text_duplicate(
    session: Session, workspace_id: str, text_sha256: str, exclude_version_id: str
) -> DocumentVersion | None:
    return session.scalar(
        select(DocumentVersion)
        .where(
            DocumentVersion.workspace_id == workspace_id,
            DocumentVersion.text_sha256 == text_sha256,
            DocumentVersion.id != exclude_version_id,
        )
        .limit(1)
    )


def enqueue_job(session: Session, workspace_id: str, kind: str, document_version_id: str | None = None) -> None:
    session.add(
        IngestionJob(
            id=new_id("job"),
            workspace_id=workspace_id,
            document_version_id=document_version_id,
            kind=kind,
            status="queued",
        )
    )


def enqueue_lexical_rebuild(session: Session, workspace_id: str) -> None:
    """Idempotent: a queued rebuild for this workspace absorbs the request."""
    statement = (
        insert(IngestionJob)
        .values(id=new_id("job"), workspace_id=workspace_id, kind="reindex_lexical", status="queued", attempts=0)
        .on_conflict_do_nothing(
            index_elements=["workspace_id"],
            index_where=text("kind = 'reindex_lexical' AND status = 'queued'"),
        )
    )
    session.execute(statement)


def claim_next_job(session: Session) -> IngestionJob | None:
    job = session.scalar(
        select(IngestionJob)
        .where(IngestionJob.status == "queued")
        .order_by(IngestionJob.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if job is None:
        return None
    job.status = "running"
    job.started_at = utcnow()
    job.attempts += 1
    return job


def requeue_stale_jobs(session: Session, lease: timedelta = timedelta(minutes=30)) -> int:
    """Jobs left `running` by a crashed worker go back to the queue once their lease expires."""
    result = session.execute(
        update(IngestionJob)
        .where(IngestionJob.status == "running", IngestionJob.started_at < utcnow() - lease)
        .values(status="queued")
    )
    return int(result.rowcount)  # type: ignore[attr-defined]


def add_temporal_fact(
    session: Session,
    version_id: str,
    *,
    kind: str,
    value: date,
    precision: str,
    source: str,
    confidence: str,
    evidence: str | None,
) -> None:
    session.add(
        TemporalFact(
            document_version_id=version_id,
            kind=kind,
            value=value,
            precision=precision,
            source=source,
            confidence=confidence,
            evidence=evidence,
        )
    )


def _version_sort_key(version: DocumentVersion) -> date | None:
    dates = [f.value for f in version.temporal_facts if f.kind in ("revised", "published") and f.precision == "day"]
    return max(dates) if dates else None


def recompute_latest_versions(session: Session, document_id: str) -> list[DocumentVersion]:
    """Set `is_latest_version` from ordinals, else day-precision dates. Never uses the ingestion time.

    If neither identifies an order for every version, latest is unknown and the flag is NULL.
    """
    versions = list(
        session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id, DocumentVersion.ingestion_status != "failed")
            .options(joinedload(DocumentVersion.temporal_facts))
        ).unique()
    )
    if all(v.version_ordinal is not None for v in versions) and versions:
        top = max(v.version_ordinal for v in versions if v.version_ordinal is not None)
        for version in versions:
            version.is_latest_version = version.version_ordinal == top
    elif len(versions) > 1 and all(_version_sort_key(v) is not None for v in versions):
        keys = {v.id: _version_sort_key(v) for v in versions}
        newest = max(k for k in keys.values() if k is not None)
        winners = [vid for vid, key in keys.items() if key == newest]
        for version in versions:
            # Two versions on the same day cannot be ordered; that stays unknown.
            version.is_latest_version = None if len(winners) > 1 else version.id == winners[0]
    else:
        for version in versions:
            version.is_latest_version = None
    return versions


def recompute_supersedes(session: Session, workspace_id: str, document_id: str) -> None:
    versions = sorted(
        (
            v
            for v in session.scalars(select(DocumentVersion).where(DocumentVersion.document_id == document_id))
            if v.version_ordinal is not None and v.ingestion_status != "failed"
        ),
        key=lambda v: v.version_ordinal or 0,
    )
    for older, newer in itertools.pairwise(versions):
        statement = (
            insert(DocumentRelation)
            .values(
                workspace_id=workspace_id,
                from_version_id=newer.id,
                to_version_id=older.id,
                type="supersedes",
                source="source_metadata",
            )
            .on_conflict_do_nothing(constraint="uq_relation")
        )
        session.execute(statement)


def recompute_classification(session: Session, workspace_id: str) -> str:
    """Decision D1: public only if the workspace has documents and every one was verified against the allowlist."""
    total = session.scalar(
        select(func.count()).select_from(DocumentVersion).where(DocumentVersion.workspace_id == workspace_id)
    )
    unverified = session.scalar(
        select(func.count())
        .select_from(DocumentVersion)
        .where(
            DocumentVersion.workspace_id == workspace_id,
            DocumentVersion.public_corpus_verified.is_(False),
        )
    )
    classification = "public" if total and not unverified else "private"
    session.execute(update(Workspace).where(Workspace.id == workspace_id).values(data_classification=classification))
    return classification


def published_dates(session: Session, version_ids: Sequence[str]) -> dict[str, date]:
    if not version_ids:
        return {}
    rows = session.execute(
        select(TemporalFact.document_version_id, TemporalFact.value, TemporalFact.source).where(
            TemporalFact.document_version_id.in_(version_ids), TemporalFact.kind == "published"
        )
    ).all()
    result: dict[str, date] = {}
    for version_id, value, source in rows:
        if source == "source_metadata" or version_id not in result:
            result[version_id] = value
    return result


@dataclass(frozen=True)
class LexicalRow:
    payload: ChunkPayload
    text: str


def chunks_for_lexical_index(session: Session, workspace_id: str) -> tuple[list[LexicalRow], list[str]]:
    """All indexable chunks in stable order, and the version ids this snapshot covers."""
    versions = list(
        session.scalars(
            select(DocumentVersion).where(
                DocumentVersion.workspace_id == workspace_id,
                DocumentVersion.ingestion_status.in_(READY_STATUSES),
            )
        )
    )
    version_by_id = {v.id: v for v in versions}
    published = published_dates(session, list(version_by_id))
    rows = session.scalars(
        select(Chunk)
        .where(Chunk.workspace_id == workspace_id, Chunk.document_version_id.in_(list(version_by_id)))
        .order_by(Chunk.id)
    )
    lexical_rows: list[LexicalRow] = []
    for chunk in rows:
        version = version_by_id[chunk.document_version_id]
        day = published.get(chunk.document_version_id)
        lexical_rows.append(
            LexicalRow(
                payload=ChunkPayload(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_version_id=chunk.document_version_id,
                    is_latest_version=version.is_latest_version,
                    published_at=day.isoformat() if day else None,
                    section_kind=chunk.section_kind,
                ),
                text=chunk.text,
            )
        )
    return lexical_rows, list(version_by_id)


def mark_versions_ready(session: Session, version_ids: Iterable[str]) -> None:
    session.execute(
        update(DocumentVersion)
        .where(DocumentVersion.id.in_(list(version_ids)), DocumentVersion.ingestion_status == "indexed_dense")
        .values(ingestion_status="ready")
    )


class PostgresChunkStore:
    """ChunkStore backed by Postgres. Workspace scoping is in the WHERE clause, not left to the caller."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_many(self, workspace_id: str, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        rows = self._session.execute(
            select(Chunk, DocumentVersion, Document)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.workspace_id == workspace_id, Chunk.id.in_(chunk_ids))
        ).all()
        return {
            chunk.id: ChunkRecord(
                chunk_id=chunk.id,
                workspace_id=chunk.workspace_id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                text=chunk.text,
                text_sha256=chunk.text_sha256,
                embed_text=chunk.text,
                ordinal=chunk.ordinal,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=tuple(chunk.section_path),
                section_kind=chunk.section_kind,
                is_latest_version=version.is_latest_version,
                document_title=document.title,
                version_label=version.version_label,
            )
            for chunk, version, document in rows
        }
