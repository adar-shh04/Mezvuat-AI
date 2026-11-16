"""Background worker: claims jobs from the Postgres queue. Same codebase and image as the API."""

from __future__ import annotations

import logging
import signal
import socket
import threading
import time
from types import FrameType

import httpx

from mezvuat.core.clock import utcnow
from mezvuat.core.config import Settings, get_settings
from mezvuat.core.errors import MezvuatError, ValidationFailedError
from mezvuat.core.logging import configure_logging
from mezvuat.inference import modelmesh_ops
from mezvuat.ingestion.pipeline import (
    IngestionContext,
    ingest_version,
    mark_version_failed,
    rebuild_lexical_index,
)
from mezvuat.ml.models import HfTokenCounter, SentenceTransformerEmbedder
from mezvuat.ml.registry import get_embedder_spec
from mezvuat.persistence.blobstore import LocalBlobStore
from mezvuat.persistence.db import get_session_factory
from mezvuat.persistence.modelmesh_outbox import drain_outbox
from mezvuat.persistence.models import IngestionJob, Workspace
from mezvuat.persistence.qdrant import QdrantDenseIndex, collection_name
from mezvuat.persistence.repositories import claim_next_job, recompute_classification, requeue_stale_jobs
from mezvuat.retrieval.lexical import WorkspaceLexicalIndexStore

log = logging.getLogger("mezvuat.worker")
_stop = threading.Event()


def _handle_signal(signum: int, frame: FrameType | None) -> None:
    _stop.set()


def build_context(settings: Settings) -> IngestionContext:
    spec = get_embedder_spec(settings.embedding_model_id)
    embedder = SentenceTransformerEmbedder(spec, device=settings.ml_device)
    factory = get_session_factory()

    def current_version(workspace_id: str) -> int | None:
        with factory() as session:
            workspace = session.get(Workspace, workspace_id)
            return workspace.lexical_index_version if workspace else None

    return IngestionContext(
        settings=settings,
        blobs=LocalBlobStore(settings.blob_dir),
        dense=QdrantDenseIndex.from_settings(
            settings, collection_name(spec.model_id, spec.dim, settings.index_version)
        ),
        lexical=WorkspaceLexicalIndexStore(settings.lexical_dir, current_version),
        embedder=embedder,
        token_counter=HfTokenCounter(spec.model_id, spec.revision),
        http_client=httpx.Client(),
    )


def _finish_job(job: IngestionJob, status: str, code: str | None, message: str | None) -> None:
    job.status = status
    job.error_code = code
    job.error_message = message[:2000] if message else None
    job.finished_at = utcnow()


def run_one(ctx: IngestionContext) -> bool:
    """Claim and run a single job. Returns False when the queue is empty."""
    factory = get_session_factory()
    with factory() as session:
        job = claim_next_job(session)
        if job is None:
            session.rollback()
            return False
        session.commit()
        job_id, kind, version_id, workspace_id, attempts = (
            job.id,
            job.kind,
            job.document_version_id,
            job.workspace_id,
            job.attempts,
        )
    with factory() as session:
        try:
            if kind == "ingest":
                if version_id is None:
                    raise ValidationFailedError("ingest job without a version", code="missing_source")
                ingest_version(session, ctx, version_id)
            else:
                rebuild_lexical_index(session, ctx, workspace_id)
            done = session.get_one(IngestionJob, job_id)
            _finish_job(done, "succeeded", None, None)
            session.commit()
        except ValidationFailedError as exc:
            session.rollback()
            log.warning("job failed permanently", extra={"job_id": job_id, "code": exc.code})
            if version_id:
                mark_version_failed(session, version_id, exc.code, exc.message)
            _finish_job(session.get_one(IngestionJob, job_id), "failed", exc.code, exc.message)
            session.commit()
        except Exception as exc:
            session.rollback()
            code = exc.code if isinstance(exc, MezvuatError) else "internal_error"
            log.exception("job error", extra={"job_id": job_id, "attempts": attempts})
            job_row = session.get_one(IngestionJob, job_id)
            if attempts >= ctx.settings.job_max_attempts:
                if version_id:
                    mark_version_failed(session, version_id, code, str(exc))
                else:
                    recompute_classification(session, workspace_id)
                _finish_job(job_row, "failed", code, str(exc))
            else:
                job_row.status = "queued"
            session.commit()
    return True


def main() -> None:
    settings = get_settings()
    configure_logging(settings.mezvuat_log_level)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    log.info("worker starting", extra={"host": socket.gethostname()})
    ctx = build_context(settings)
    factory = get_session_factory()
    with factory() as session:
        requeue_stale_jobs(session)
        session.commit()
    # ModelMesh side effects (M8): drained between jobs, at most every 30 s, and with short retries so a down
    # ModelMesh never stalls ingestion.
    ops = modelmesh_ops(settings)
    if ops is not None:
        ops.max_retries, ops.max_wait_s = 1, 5.0
    next_drain = 0.0
    while not _stop.is_set():
        if ops is not None and time.monotonic() >= next_drain:
            try:
                drain_outbox(factory, ops)
            except Exception:
                log.exception("ModelMesh outbox drain failed")
            next_drain = time.monotonic() + 30.0
        worked = run_one(ctx)
        if not worked:
            _stop.wait(settings.worker_poll_interval_s)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
