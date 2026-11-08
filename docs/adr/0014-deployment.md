# 0014. Deployment (local Docker Compose; cloud deferred)

Proposed · 2026-11-08 · Spec §5.1, §5.2, §6 ("runs independently using Docker"), §8 item 10

## Context

- Mezvuat must run on its own in Docker on a 15.7 GB, 4 GB-VRAM Windows laptop with Docker Desktop (WSL2).
- ModelMesh is a separate deployable that Mezvuat reaches over HTTP.
- Langfuse self-host (v4.x) needs web, worker, Postgres, ClickHouse, Redis/Valkey and S3/MinIO. Its docs recommend at least 4 cores and 16 GiB for the compose stack.
- Langfuse Cloud's Hobby tier is free:
  - 50k units per month;
  - 30 days of data access;
  - 2 users.

## Options considered

**Langfuse:**
- (a) Self-host in the same compose file. Too heavy for this machine next to everything else.
- (b) **Langfuse Cloud Hobby**, with content masking on by default. Opt-in.
- (c) No Langfuse: local JSON logs plus the Postgres `llm_calls` table. Always available as the baseline.

**Raw files:** MinIO vs a local volume. Local volume chosen (ADR-0003).

**Cloud:** deferred. There is no requirement or budget yet, and the spec warns against unnecessary infrastructure.

## Decision

`infra/compose.yaml` with profiles:

| Service | Image | Profile | Notes |
|---|---|---|---|
| `postgres` | `postgres:17` (exact tag pinned at M1; pg_textsearch-compatible if needed later) | default | App DB plus the `mlflow` database. Healthcheck. Volume. |
| `qdrant` | `qdrant/qdrant:v1.20.x` | default | Volume. Ports 6333/6334 on localhost only. |
| `api` | `mezvuat-backend` (python:3.13-slim, CPU torch) | default | FastAPI with uvicorn. Runs migrations on start through a separate `migrate` one-shot service. |
| `worker` | same image | default | Ingestion, embedding, BM25 rebuilds and eval jobs (Postgres queue) |
| `frontend` | `mezvuat-frontend` (node:24-alpine, `next build` + `next start`) | default | |
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.18.0` (plus psycopg if the image lacks it; checked at M6) | `ml` | Postgres backend and artifact volume |
| `grobid` | `grobid/grobid:0.9.1-crf` | `grobid` | Optional section and metadata enricher (about 4 GB RAM) |
| `api-gpu` | backend image with CUDA torch | `gpu` | Only if Docker Desktop GPU passthrough is verified. Otherwise ML jobs run natively via uv. |

Further rules:

- **Langfuse is external and optional.** If `LANGFUSE_HOST` and its keys are set, traces go there (Cloud Hobby by default), with content masking **on**. Self-hosting is documented as an alternative for a machine with ≥ 16 GiB free.
  - **Open question for the product owner:** is sending masked traces to Langfuse Cloud acceptable?
- **ModelMesh is external.** It is reached by `MODELMESH_BASE_URL`, for example `http://host.docker.internal:<port>`. The two products share no compose file and no network. The integration test brings both up independently.
- **Networking.** Only `frontend` (3000) and `api` (8000) are published, on 127.0.0.1. All secrets come from `.env` (`.env.example` lists the names), and nothing is baked into images.
- **Resources.**
  - Qdrant, Postgres and api/worker get memory limits, for an expected total of about 5–6 GB (EST, see ENVIRONMENT.md).
  - The `HF_HOME` model cache is a named volume, so models download once.
- **CI:** GitHub Actions on ubuntu with Python 3.13 and CPU torch.
  - Jobs: ruff, mypy, import-linter, unit tests.
  - Integration tests run against service containers (Postgres, Qdrant).
  - The frontend is linted and built.
- **Cloud:** deferred. When it is needed, a single VM running the same compose file is the first step, before any managed services. That would be its own ADR.

## Consequences

- One `docker compose up` gives a working Mezvuat with no keys at all for ingestion and retrieval. LLM features need `GROQ_API_KEY` or ModelMesh.
- Langfuse Cloud adds an external data flow. Masking reduces the risk, but this needs a product-owner decision.
- CPU-only containers make in-container embedding slower, so bulk indexing can run natively on the GPU via the CLI, writing to the same Qdrant.

## Verification plan

- `docker compose up -d`, then `/health` reports Postgres and Qdrant `ok`, then an ingest of a sample PDF, then a retrieval query returns cited chunks. Scripted as `make smoke` or an equivalent PowerShell/Python script.
- Bring the stack up with Langfuse variables unset and confirm everything except tracing works.
- `docker stats` snapshot recorded in docs to confirm the memory estimate.

## Sources (accessed 2026-11-08)

- langfuse.com/self-hosting
- langfuse.com/self-hosting/deployment/docker-compose
- langfuse.com/pricing
- github.com/mlflow/mlflow/pkgs/container/mlflow
- grobid.readthedocs.io/en/latest/Grobid-docker/
