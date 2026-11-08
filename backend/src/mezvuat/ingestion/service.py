"""Request-side ingestion: validate, store the raw bytes, create the version and queue the work."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mezvuat.core.config import Settings
from mezvuat.core.errors import NotFoundError
from mezvuat.core.ids import new_id
from mezvuat.ingestion.corpus.manifest import ManifestVersion, cached_allowlist
from mezvuat.ingestion.ssrf import validate_url
from mezvuat.ingestion.validation import SOURCE_MEDIA_TYPES, detect_source_type, validate_pdf, validate_size
from mezvuat.persistence.blobstore import BlobStore
from mezvuat.persistence.models import Document, DocumentVersion, Workspace
from mezvuat.persistence.repositories import (
    add_temporal_fact,
    enqueue_job,
    find_document_by_key,
    find_version_by_raw,
    recompute_classification,
)


@dataclass(frozen=True)
class SubmitResult:
    version: DocumentVersion
    duplicate: bool


def _require_workspace(session: Session, workspace_id: str) -> Workspace:
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise NotFoundError("workspace not found")
    return workspace


def _apply_corpus_metadata(session: Session, version: DocumentVersion, entry: ManifestVersion) -> None:
    """Facts for an allowlisted arXiv version come from the manifest, not the caller (ADR-0013)."""
    version.public_corpus_verified = True
    version.version_label = entry.label
    version.version_ordinal = entry.ordinal
    version.license = entry.license
    version.source_uri = f"https://arxiv.org/abs/{entry.arxiv_id}{entry.label}"
    add_temporal_fact(
        session,
        version.id,
        kind="published",
        value=entry.first_submitted,
        precision="day",
        source="source_metadata",
        confidence="high",
        evidence="arXiv OAI-PMH arXivRaw: v1 submission date",
    )
    if entry.ordinal > 1:
        add_temporal_fact(
            session,
            version.id,
            kind="revised",
            value=entry.submitted,
            precision="day",
            source="source_metadata",
            confidence="high",
            evidence=f"arXiv OAI-PMH arXivRaw: {entry.label} submission date",
        )


def submit_file(
    session: Session,
    blobs: BlobStore,
    settings: Settings,
    workspace_id: str,
    raw: bytes,
    filename: str | None,
) -> SubmitResult:
    _require_workspace(session, workspace_id)
    validate_size(raw, settings.upload_max_bytes)
    source_type = detect_source_type(raw, filename)
    if source_type == "pdf":
        validate_pdf(raw, settings.pdf_max_pages)
    digest = blobs.put(raw)
    existing = find_version_by_raw(session, workspace_id, digest)
    if existing is not None:
        return SubmitResult(existing, duplicate=True)

    entry = cached_allowlist(settings.mezvuat_corpus_manifest).get(digest)
    document: Document | None = None
    if entry is not None:
        document = find_document_by_key(session, workspace_id, f"arxiv:{entry.arxiv_id}")
    if document is None:
        document = Document(
            id=new_id("doc"),
            workspace_id=workspace_id,
            canonical_key=f"arxiv:{entry.arxiv_id}" if entry else None,
            title=entry.title if entry else filename,
            authors=[entry.authors] if entry else None,
        )
        session.add(document)
        session.flush()
    version = DocumentVersion(
        id=new_id("ver"),
        document_id=document.id,
        workspace_id=workspace_id,
        raw_sha256=digest,
        source_type=source_type,
        original_filename=filename,
        media_type=SOURCE_MEDIA_TYPES[source_type],
        size_bytes=len(raw),
    )
    session.add(version)
    try:
        session.flush()
    except IntegrityError:
        # A concurrent upload of the same bytes won the race.
        session.rollback()
        winner = find_version_by_raw(session, workspace_id, digest)
        if winner is None:
            raise
        return SubmitResult(winner, duplicate=True)
    if entry is not None:
        _apply_corpus_metadata(session, version, entry)
    enqueue_job(session, workspace_id, "ingest", version.id)
    recompute_classification(session, workspace_id)
    session.commit()
    return SubmitResult(version, duplicate=False)


def submit_url(session: Session, workspace_id: str, url: str) -> SubmitResult:
    """Reject bad URLs immediately; the worker fetches, re-validating every hop."""
    _require_workspace(session, workspace_id)
    validate_url(url)
    document = Document(id=new_id("doc"), workspace_id=workspace_id, title=url)
    session.add(document)
    session.flush()
    version = DocumentVersion(
        id=new_id("ver"),
        document_id=document.id,
        workspace_id=workspace_id,
        source_type="url",
        source_uri=url,
    )
    session.add(version)
    session.flush()
    enqueue_job(session, workspace_id, "ingest", version.id)
    recompute_classification(session, workspace_id)
    session.commit()
    return SubmitResult(version, duplicate=False)
