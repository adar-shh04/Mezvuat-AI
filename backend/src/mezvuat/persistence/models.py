"""SQLAlchemy models. Postgres is the source of truth; Qdrant and BM25 indices can be rebuilt from it."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column, relationship

from mezvuat.core.clock import utcnow


class Base(DeclarativeBase):
    pass


class Workspace(Base):
    __tablename__ = "workspace"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    # Derived by the system (decision D1). Never settable through the API.
    data_classification: Mapped[str] = mapped_column(String(16), server_default="private")
    lexical_index_version: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    documents: Mapped[list[Document]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("data_classification IN ('private', 'public')", name="ck_workspace_classification"),
    )


class Document(Base):
    """A logical work, e.g. arXiv 2401.01234 across all its versions."""

    __tablename__ = "document"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    canonical_key: Mapped[str | None] = mapped_column(String(300))
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str] | None] = mapped_column(JSONB)
    issuing_org: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    workspace: Mapped[Workspace] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index(
            "uq_document_canonical",
            "workspace_id",
            "canonical_key",
            unique=True,
            postgresql_where=text("canonical_key IS NOT NULL"),
        ),
    )


class DocumentVersion(Base):
    """One concrete content state of a document."""

    __tablename__ = "document_version"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    version_label: Mapped[str | None] = mapped_column(String(64))
    version_ordinal: Mapped[int | None] = mapped_column(Integer)
    raw_sha256: Mapped[str | None] = mapped_column(String(64))
    text_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    probable_duplicate_of: Mapped[str | None] = mapped_column(String(40))
    source_type: Mapped[str] = mapped_column(String(16))
    source_uri: Mapped[str | None] = mapped_column(Text)
    original_filename: Mapped[str | None] = mapped_column(Text)
    media_type: Mapped[str | None] = mapped_column(String(100))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    license: Mapped[str | None] = mapped_column(Text)
    public_corpus_verified: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    is_latest_version: Mapped[bool | None] = mapped_column(Boolean)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # pending -> processing -> indexed_dense -> ready | failed
    ingestion_status: Mapped[str] = mapped_column(String(20), server_default="pending")
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    extractor: Mapped[str | None] = mapped_column(String(64))
    chunker_version: Mapped[str | None] = mapped_column(String(32))
    normalized_text: Mapped[str | None] = deferred(mapped_column(Text))
    pages: Mapped[list[dict[str, int]] | None] = deferred(mapped_column(JSONB))
    sections: Mapped[list[dict[str, Any]] | None] = deferred(mapped_column(JSONB))

    document: Mapped[Document] = relationship(back_populates="versions")
    temporal_facts: Mapped[list[TemporalFact]] = relationship(
        back_populates="version", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        # NULL raw_sha256 (a URL not fetched yet) never collides; Postgres treats NULLs as distinct.
        UniqueConstraint("workspace_id", "raw_sha256", name="uq_version_workspace_raw"),
        CheckConstraint(
            "ingestion_status IN ('pending', 'processing', 'indexed_dense', 'ready', 'failed')",
            name="ck_version_status",
        ),
    )


class TemporalFact(Base):
    """One dated fact. A missing row means the date is unknown; dates are never substituted for each other."""

    __tablename__ = "temporal_fact"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_version.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    value: Mapped[date] = mapped_column(Date)
    precision: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(24))
    confidence: Mapped[str] = mapped_column(String(8))
    evidence: Mapped[str | None] = mapped_column(Text)

    version: Mapped[DocumentVersion] = relationship(back_populates="temporal_facts")

    __table_args__ = (
        CheckConstraint("kind IN ('published', 'revised', 'effective_from', 'effective_to')", name="ck_temporal_kind"),
        CheckConstraint("precision IN ('day', 'month', 'year')", name="ck_temporal_precision"),
        CheckConstraint("confidence IN ('high', 'medium', 'low')", name="ck_temporal_confidence"),
        UniqueConstraint("document_version_id", "kind", "source", name="uq_temporal_version_kind_source"),
    )


class DocumentRelation(Base):
    __tablename__ = "document_relation"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    from_version_id: Mapped[str] = mapped_column(ForeignKey("document_version.id", ondelete="CASCADE"))
    to_version_id: Mapped[str | None] = mapped_column(ForeignKey("document_version.id", ondelete="CASCADE"))
    to_document_id: Mapped[str | None] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(24))
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint(
            "type IN ('version_of', 'supersedes', 'cites', 'amends', 'contradicts')", name="ck_relation_type"
        ),
        UniqueConstraint("from_version_id", "to_version_id", "type", name="uq_relation"),
    )


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"))
    document_version_id: Mapped[str] = mapped_column(ForeignKey("document_version.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[list[str]] = mapped_column(JSONB)
    section_kind: Mapped[str] = mapped_column(String(16))
    token_count: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String(64))
    chunker_version: Mapped[str] = mapped_column(String(32))

    __table_args__ = (
        UniqueConstraint("document_version_id", "ordinal", "chunker_version", name="uq_chunk_position"),
        CheckConstraint("char_end > char_start", name="ck_chunk_span"),
    )


class IngestionJob(Base):
    """Postgres-backed work queue (FOR UPDATE SKIP LOCKED); see ADR-0001."""

    __tablename__ = "ingestion_job"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    document_version_id: Mapped[str | None] = mapped_column(ForeignKey("document_version.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(12), server_default="queued")
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("kind IN ('ingest', 'reindex_lexical')", name="ck_job_kind"),
        CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')", name="ck_job_status"),
        Index("ix_job_claim", "status", "created_at"),
        # At most one queued lexical rebuild per workspace: ingest bursts coalesce into one rebuild.
        Index(
            "uq_job_one_queued_reindex",
            "workspace_id",
            unique=True,
            postgresql_where=text("kind = 'reindex_lexical' AND status = 'queued'"),
        ),
    )


class LlmCall(Base):
    """System of record for joining research runs to LLM executions (ADR-0012). Populated from M4."""

    __tablename__ = "llm_call"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("workspace.id", ondelete="SET NULL"), index=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    workflow_node: Mapped[str | None] = mapped_column(String(64))
    task_type: Mapped[str] = mapped_column(String(32))
    backend: Mapped[str] = mapped_column(String(16))
    model: Mapped[str | None] = mapped_column(String(200))
    client_request_id: Mapped[str] = mapped_column(String(64), unique=True)
    modelmesh_request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    modelmesh_trace_id: Mapped[str | None] = mapped_column(String(32))
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(64))
    fallback_reason: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    usage_source: Mapped[str | None] = mapped_column(String(24))
    cost_estimated_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cache_status: Mapped[str | None] = mapped_column(String(16))
    # Added in M4 (migration 0002).
    provider: Mapped[str | None] = mapped_column(String(32))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    iteration: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(16))
    error_code: Mapped[str | None] = mapped_column(String(64))
    http_status: Mapped[int | None] = mapped_column(Integer)
    finish_reason: Mapped[str | None] = mapped_column(String(16))
    structured_mode: Mapped[str | None] = mapped_column(String(24))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ResearchRun(Base):
    """One research question and its outcome. The step-by-step state lives in the LangGraph checkpoint tables."""

    __tablename__ = "research_run"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), server_default="queued")
    data_policy: Mapped[str] = mapped_column(String(16))
    retrieval_mode: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name="ck_run_status"),
        CheckConstraint("data_policy IN ('any', 'local_only')", name="ck_run_data_policy"),
    )


class RunFeedback(Base):
    """A user's thumbs up or down on a run's answer, or on one claim in it (M7). M6 forwards these to Langfuse."""

    __tablename__ = "run_feedback"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_run.id", ondelete="CASCADE"), index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspace.id", ondelete="CASCADE"), index=True)
    rating: Mapped[str] = mapped_column(String(8))
    claim_id: Mapped[str | None] = mapped_column(String(40))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (CheckConstraint("rating IN ('up', 'down')", name="ck_feedback_rating"),)


class ModelMeshOutbox(Base):
    """Durable queue of ModelMesh side effects (M8): feedback batches, and the cache purge and feedback deletion a
    workspace deletion requires. No foreign key on purpose: the rows must outlive the workspace they clean up."""

    __tablename__ = "modelmesh_outbox"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(12), server_default="queued")
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("kind IN ('cache_invalidate', 'feedback_delete', 'feedback_post')", name="ck_outbox_kind"),
        CheckConstraint("status IN ('queued', 'done', 'failed')", name="ck_outbox_status"),
        Index("ix_outbox_claim", "status", "created_at"),
    )
