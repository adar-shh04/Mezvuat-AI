# 0001. Repository layout and module boundaries

Proposed · 2026-11-08 · Spec §1.5, §5.1, §5.2

## Context

The spec asks for a modular monolith, ten layers with clean boundaries, and no microservices "until a workload justifies them". It also asks for an independent repository with its own Docker setup and CI. The ML code (training and evaluation) must be reproducible, but it should not bloat the API's runtime image.

## Options considered

1. **One Python package with enforced internal module boundaries, plus a separate `frontend/` Next.js app.**
2. Several Python distributions in a uv workspace, one per layer. This gives strong isolation, but packaging overhead lands on day one.
3. Microservices (an ingestion service, a retrieval service, and so on). The spec rules this out.

## Decision

Option 1. The two top-level deployables are `backend/` (Python) and `frontend/` (Next.js). The backend layout looks like this:

```
backend/
  pyproject.toml            # one distribution "mezvuat"; optional dependency groups: ml, eval, dev
  src/mezvuat/
    core/            # config (pydantic-settings), ids (ULID), errors, logging, clock; no deps on other layers
    api/             # FastAPI routers, request/response schemas, auth, workspace scoping   (§1.5 API)
    workspaces/      # workspace domain service
    ingestion/       # validate → extract → normalize → chunk → enqueue embedding/indexing (§1.5 Ingestion)
      parsers/       # pdf, text/markdown, url (pluggable Parser protocol)
    retrieval/       # lexical (BM25), dense, fusion, filters, rerank, Retriever pipeline (§1.5 Retrieval)
    orchestration/   # LangGraph graph, nodes, state, prompts                                 (§1.5 Orchestration)
    citations/       # evidence registry, span checks, claim verification                    (part of Orchestration, split out for testability)
    temporal/        # version/date model and as-of resolution                               (§1.3E)
    ml/              # Embedder/Reranker protocols + sentence-transformers impls, training    (§1.5 ML)
    inference/       # LLMClient protocol, modelmesh/ and direct/ backends                    (§1.5 Inference)
    evaluation/      # retrieval metrics, RAGAS adapters, custom metrics, feedback export    (§1.5 Evaluation)
    persistence/     # SQLAlchemy models, repositories, Alembic, Qdrant adapter, blob store  (§1.5 Persistence)
    observability/   # structured logging, Langfuse adapter, request-id propagation          (§1.5 Observability)
    worker/          # background job runner (same codebase, separate process)
  migrations/        # Alembic
  tests/{unit,integration,eval}/
experiments/         # versioned experiment configs (YAML) + CLI entrypoints; results go to MLflow, not git
data/                # git-ignored: raw files, BEIR downloads, model cache; only manifests are committed
frontend/            # Next.js app (ADR-0002)
infra/               # docker compose files, service configs
docs/                # spec, contract mirror, ADRs, plan
```

The boundary rules below are enforced in CI with `import-linter`, a small dev-only dependency whose job is exactly this. Its version is checked at M1.

- `core` imports nothing from the other layers.
- Only `persistence` talks to Postgres, Qdrant or the blob store. Other layers depend on the repository protocols defined there.
- `orchestration` uses `retrieval`, `citations`, `temporal` and `inference` only through their public `__init__` interfaces.
- Only `inference` makes HTTP calls to LLM providers or ModelMesh. Nothing else imports an LLM SDK or knows provider names.
- `ml` training code is never imported by `api`. The training entrypoints live under `experiments/` and `ml/training/`.
- `evaluation` may import anything. Nothing on the request path imports `evaluation`, except the online feedback recorder.

The worker runs from the same image as the API with a different command. Jobs are queued in a Postgres table (`SELECT … FOR UPDATE SKIP LOCKED`). This avoids adding Redis or Celery to Mezvuat: the spec says Redis is for ephemeral cache and rate-limit state, and Mezvuat needs neither.

## Consequences

- **Good:**
  - Everything lives in one place.
  - Refactoring a boundary is cheap.
  - Retrieval runs as a library, so ML experiments can call the exact production pipeline.
- **Bad:**
  - One Python environment means heavy optional groups (`ml` with torch, `eval` with ragas) must be kept out of the API image by careful dependency grouping.
- **Splitting out a service later**, for example a GPU embedding worker, only needs a new entrypoint, because the protocols already exist.

## Verification plan

- CI runs `lint-imports` with the contracts above, and a violation fails the build.
- A unit test asserts that `mezvuat.api` imports no module under `mezvuat.ml.training` or `mezvuat.evaluation` (except the feedback recorder).
- The API Docker image build installs no `eval` group; a CI check confirms `ragas` is absent from that image.
