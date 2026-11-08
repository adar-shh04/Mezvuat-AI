"""End-to-end ingestion and retrieval against real Postgres, Qdrant and models, with the real worker code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from mezvuat.api.deps import get_blob_store, get_dense_index
from mezvuat.core.config import get_settings
from mezvuat.core.errors import RetrievalUnavailableError
from mezvuat.ingestion.corpus.manifest import cached_allowlist
from mezvuat.ingestion.pipeline import IngestionContext
from mezvuat.persistence.db import session_scope
from mezvuat.persistence.models import Chunk, DocumentRelation, DocumentVersion, IngestionJob
from mezvuat.worker.main import build_context, run_one

pytestmark = [pytest.mark.integration, pytest.mark.live_model]

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
SENTENCE_P1 = "Quantum annealing schedules outperform greedy baselines on sparse retrieval graphs."
SENTENCE_P2 = "The zirconium catalyst degrades after nine hundred reaction cycles at high pressure."


def make_pdf(marker: str = "") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "1 Introduction", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 90, 520, 300), f"{SENTENCE_P1} {marker}", fontsize=11)
    page = document.new_page()
    page.insert_text((72, 72), "2 Materials", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 90, 520, 300), SENTENCE_P2, fontsize=11)
    page.insert_text((72, 330), "References", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 350, 520, 500), "[1] A. Author. Zirconium cited work. 2020.", fontsize=11)
    data: bytes = document.tobytes()
    document.close()
    return data


@pytest.fixture(scope="module")
def ingestion_context() -> Iterator[IngestionContext]:
    ctx = build_context(get_settings())
    yield ctx
    ctx.http_client.close()


def drain(ctx: IngestionContext) -> None:
    while run_one(ctx):
        pass


def create_workspace(client: TestClient, name: str = "w") -> str:
    response = client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def upload(client: TestClient, workspace_id: str, raw: bytes, name: str = "paper.pdf") -> dict[str, object]:
    response = client.post(
        f"/api/v1/workspaces/{workspace_id}/documents", files={"file": (name, raw, "application/pdf")}
    )
    assert response.status_code in (200, 202), response.text
    return dict(response.json())


def search(
    client: TestClient, workspace_id: str, query: str, mode: str = "hybrid_rrf", **extra: object
) -> dict[str, object]:
    response = client.post(f"/api/v1/workspaces/{workspace_id}/search", json={"query": query, "mode": mode, **extra})
    assert response.status_code == 200, response.text
    return dict(response.json())


def write_allowlist(entries: list[tuple[bytes, str, int]]) -> None:
    """One paper with the given (pdf bytes, label, ordinal) versions."""
    manifest = {
        "papers": [
            {
                "arxiv_id": "2401.12345",
                "title": "Allowlisted paper",
                "authors": "A. Author",
                "license": CC_BY,
                "versions": [
                    {
                        "label": label,
                        "ordinal": ordinal,
                        "submitted": f"2024-0{ordinal}-15",
                        "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                    for raw, label, ordinal in entries
                ],
            }
        ]
    }
    Path(get_settings().mezvuat_corpus_manifest).write_text(json.dumps(manifest), encoding="utf-8")
    cached_allowlist.cache_clear()


def test_full_pipeline_preserves_provenance(client: TestClient, ingestion_context: IngestionContext) -> None:
    workspace = create_workspace(client)
    accepted = upload(client, workspace, make_pdf())
    assert accepted["status"] == "accepted"
    drain(ingestion_context)

    docs = client.get(f"/api/v1/workspaces/{workspace}/documents").json()
    version = docs[0]["versions"][0]
    assert version["ingestion_status"] == "ready", version
    assert version["extractor"]

    with session_scope() as session:
        row = session.get_one(DocumentVersion, version["id"])
        text = row.normalized_text
        assert text is not None
        chunks = session.scalars(select(Chunk).where(Chunk.document_version_id == row.id)).all()
        assert chunks
        for chunk in chunks:
            assert text[chunk.char_start : chunk.char_end] == chunk.text
        kinds = {tuple(c.section_path): c.section_kind for c in chunks}
    assert kinds[("References",)] == "references"
    assert ("1 Introduction",) in kinds

    for mode in ("bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"):
        result = search(client, workspace, "zirconium catalyst reaction cycles", mode=mode)
        top = result["evidence"][0]  # type: ignore[index]
        assert "zirconium catalyst" in top["text"]
        assert top["page_start"] == 2
        assert top["section_path"] == ["2 Materials"]
        assert top["section_kind"] != "references"


def test_reference_sections_are_excluded_unless_requested(
    client: TestClient, ingestion_context: IngestionContext
) -> None:
    workspace = create_workspace(client)
    upload(client, workspace, make_pdf())
    drain(ingestion_context)
    default = search(client, workspace, "Zirconium cited work", mode="bm25")
    assert all(e["section_kind"] != "references" for e in default["evidence"])  # type: ignore[attr-defined]
    included = search(client, workspace, "Zirconium cited work", mode="bm25", filters={"include_references": True})
    assert any(e["section_kind"] == "references" for e in included["evidence"])  # type: ignore[attr-defined]


def test_reingesting_identical_bytes_creates_nothing(client: TestClient, ingestion_context: IngestionContext) -> None:
    workspace = create_workspace(client)
    raw = make_pdf()
    first = upload(client, workspace, raw)
    drain(ingestion_context)
    with session_scope() as session:
        chunks_before = len(session.scalars(select(Chunk)).all())
        jobs_before = len(session.scalars(select(IngestionJob)).all())

    second = upload(client, workspace, raw)
    assert second["status"] == "duplicate"
    assert second["version"]["id"] == first["version"]["id"]  # type: ignore[index]
    drain(ingestion_context)
    with session_scope() as session:
        assert len(session.scalars(select(Chunk)).all()) == chunks_before
        assert len(session.scalars(select(IngestionJob)).all()) == jobs_before
    assert len(client.get(f"/api/v1/workspaces/{workspace}/documents").json()) == 1


def test_same_text_different_bytes_is_flagged_not_merged(
    client: TestClient, ingestion_context: IngestionContext
) -> None:
    workspace = create_workspace(client)
    text = b"Reciprocal rank fusion combines ranked lists without calibrating scores."
    upload(client, workspace, text, name="a.txt")
    upload(client, workspace, text + b"\n", name="b.txt")
    drain(ingestion_context)
    docs = client.get(f"/api/v1/workspaces/{workspace}/documents").json()
    assert len(docs) == 2
    assert sum(1 for d in docs if d["versions"][0]["probable_duplicate_of"]) == 1


def test_workspaces_are_isolated_in_both_indexes(client: TestClient, ingestion_context: IngestionContext) -> None:
    first, second = create_workspace(client, "a"), create_workspace(client, "b")
    upload(client, first, b"The osmium ledger records every confidential transfer.", name="a.txt")
    upload(client, second, b"Gardening advice for tomatoes in shaded soil.", name="b.txt")
    drain(ingestion_context)
    for mode in ("bm25", "dense", "hybrid_rrf"):
        evidence = search(client, second, "osmium ledger confidential transfer", mode=mode)["evidence"]
        assert all("osmium" not in e["text"] for e in evidence)  # type: ignore[attr-defined]


def test_workspace_classification_is_verified_and_never_client_set(
    client: TestClient, ingestion_context: IngestionContext
) -> None:
    public_pdf = make_pdf("public copy")
    write_allowlist([(public_pdf, "v1", 1)])

    rejected = client.post("/api/v1/workspaces", json={"name": "x", "data_classification": "public"})
    assert rejected.status_code == 422

    workspace = create_workspace(client)
    assert client.get(f"/api/v1/workspaces/{workspace}").json()["data_classification"] == "private"

    upload(client, workspace, public_pdf)
    drain(ingestion_context)
    assert client.get(f"/api/v1/workspaces/{workspace}").json()["data_classification"] == "public"

    upload(client, workspace, b"A private note, not on any allowlist.", name="note.txt")
    assert client.get(f"/api/v1/workspaces/{workspace}").json()["data_classification"] == "private"


def test_versions_use_ordinals_and_keep_dates_distinct(client: TestClient, ingestion_context: IngestionContext) -> None:
    v1, v2 = make_pdf("first revision"), make_pdf("second revision")
    write_allowlist([(v1, "v1", 1), (v2, "v2", 2)])
    workspace = create_workspace(client)
    # Deliberately ingest the newest version first: ingestion order must not decide "latest".
    upload(client, workspace, v2, name="v2.pdf")
    drain(ingestion_context)
    upload(client, workspace, v1, name="v1.pdf")
    drain(ingestion_context)

    docs = client.get(f"/api/v1/workspaces/{workspace}/documents").json()
    assert len(docs) == 1 and docs[0]["canonical_key"] == "arxiv:2401.12345"
    versions = {v["version_label"]: v for v in docs[0]["versions"]}
    assert versions["v2"]["is_latest_version"] is True
    assert versions["v1"]["is_latest_version"] is False
    facts = {f["kind"]: f for f in versions["v2"]["temporal_facts"]}
    assert facts["published"]["value"] == "2024-01-15"
    assert facts["revised"]["value"] == "2024-02-15"
    assert facts["published"]["source"] == "source_metadata"
    assert "revised" not in {f["kind"] for f in versions["v1"]["temporal_facts"]}

    latest = search(client, workspace, "second revision", mode="bm25", filters={"latest_only": True})["evidence"]
    assert latest and all(e["version_label"] == "v2" for e in latest)  # type: ignore[attr-defined]
    dense_latest = search(client, workspace, "second revision", mode="dense", filters={"latest_only": True})["evidence"]
    assert dense_latest and all(e["version_label"] == "v2" for e in dense_latest)  # type: ignore[attr-defined]

    with session_scope() as session:
        relation = session.scalar(select(DocumentRelation).where(DocumentRelation.type == "supersedes"))
        assert relation is not None


def test_unknown_dates_stay_unknown(client: TestClient, ingestion_context: IngestionContext) -> None:
    workspace = create_workspace(client)
    upload(client, workspace, b"A plain note with no date metadata at all.", name="n.txt")
    drain(ingestion_context)
    version = client.get(f"/api/v1/workspaces/{workspace}/documents").json()[0]["versions"][0]
    assert version["temporal_facts"] == []
    assert version["is_latest_version"] is None
    assert version["ingested_at"]


def test_invalid_uploads_get_clear_errors(client: TestClient) -> None:
    workspace = create_workspace(client)
    url = f"/api/v1/workspaces/{workspace}/documents"
    fake_pdf = client.post(url, files={"file": ("a.pdf", b"not really", "application/pdf")})
    assert fake_pdf.status_code == 415
    assert fake_pdf.json()["error"]["code"] == "content_type_mismatch"
    corrupt = client.post(url, files={"file": ("a.pdf", b"%PDF-1.7 garbage", "application/pdf")})
    assert corrupt.status_code == 422
    blocked = client.post(f"{url}/url", json={"url": "http://127.0.0.1:8080/x"})
    assert blocked.status_code == 400
    assert blocked.json()["error"]["code"] == "url_blocked_address"


def test_encrypted_pdf_is_rejected(client: TestClient) -> None:
    workspace = create_workspace(client)
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "secret")
    encrypted = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    response = client.post(
        f"/api/v1/workspaces/{workspace}/documents", files={"file": ("e.pdf", encrypted, "application/pdf")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "encrypted_pdf"


def test_pdf_without_a_text_layer_fails_visibly(client: TestClient, ingestion_context: IngestionContext) -> None:
    workspace = create_workspace(client)
    document = pymupdf.open()
    document.new_page()
    upload(client, workspace, document.tobytes())
    drain(ingestion_context)
    version = client.get(f"/api/v1/workspaces/{workspace}/documents").json()[0]["versions"][0]
    assert version["ingestion_status"] == "failed"
    assert version["error_code"] == "no_text_layer"


def test_deleting_a_workspace_removes_everything(client: TestClient, ingestion_context: IngestionContext) -> None:
    workspace, other = create_workspace(client, "gone"), create_workspace(client, "stays")
    shared, unique = make_pdf("shared"), make_pdf("unique")
    upload(client, workspace, shared)
    upload(client, workspace, unique)
    upload(client, other, shared)
    drain(ingestion_context)
    dense = get_dense_index()
    assert dense.count(workspace) > 0

    assert client.delete(f"/api/v1/workspaces/{workspace}").status_code == 204
    assert client.get(f"/api/v1/workspaces/{workspace}").status_code == 404
    assert dense.count(workspace) == 0
    assert dense.count(other) > 0
    blobs = get_blob_store()
    assert not blobs.exists(hashlib.sha256(unique).hexdigest())
    assert blobs.exists(hashlib.sha256(shared).hexdigest()), "a blob another workspace uses must survive"
    with session_scope() as session:
        assert not session.scalars(select(Chunk).where(Chunk.workspace_id == workspace)).all()


def test_qdrant_outage_reports_unavailable_and_never_answers(
    client: TestClient, ingestion_context: IngestionContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = create_workspace(client)
    upload(client, workspace, b"Some indexed content about retrieval outages.", name="a.txt")
    drain(ingestion_context)

    class DownIndex:
        def search(self, *args: object, **kwargs: object) -> list[object]:
            raise RetrievalUnavailableError("vector store unavailable: ConnectError")

    monkeypatch.setattr("mezvuat.api.routers.search.get_dense_index", DownIndex)
    response = client.post(
        f"/api/v1/workspaces/{workspace}/search", json={"query": "retrieval outages", "mode": "hybrid_rrf"}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "retrieval_unavailable"
    assert "evidence" not in response.json()


def test_health_reports_dependencies(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "postgres": "ok", "qdrant": "ok"}
    assert response.headers["X-Request-ID"]
