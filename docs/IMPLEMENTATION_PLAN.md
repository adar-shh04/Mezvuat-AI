# Mezvuat AI: implementation plan

- **Status:** architecture phase, 2026-11-08. No code exists yet.
- **Authorization:** implementation starts only when the product owner authorizes it, after reviewing the ADRs in `docs/adr/` and the contract change requests in `docs/contract/CHANGE_REQUESTS.md`.
- **Invariant:** Mezvuat is fully usable **without ModelMesh**. M1–M7 never require it, and ModelMesh appears only in M8.
- **Contract baseline:** ModelMesh Inference API **1.0-rc2** (2026-11-08). `docs/contract/openapi-v1.yaml` is normative. CR dispositions are in `docs/contract/CR_RESOLUTIONS.md`, and Mezvuat's acceptance status is in `CHANGE_REQUESTS.md`, section "Resolution status".

## Checklist keys (spec §6)

**Mezvuat (D):**

| Key | Item |
|---|---|
| D1 | Workspaces |
| D2 | Ingest/parse/index/trace |
| D3 | Hybrid retrieval |
| D4 | Reranking toggle and eval |
| D5 | Labelled retrieval dataset |
| D6 | Reproducible experiments |
| D7 | Bounded LangGraph |
| D8 | Verifiable citations |
| D9 | Temporal metadata preserved |
| D10 | App-level quality eval |
| D11 | ModelMesh and direct fallback |
| D12 | Runs in Docker |

**Combined (C):**

| Key | Item |
|---|---|
| C1 | Uses ModelMesh public API |
| C2 | Independently deployable |
| C3 | Shared request ids |
| C4 | Quality ↔ execution association |
| C5 | Workflow through ModelMesh |
| C6 | Fallback when ModelMesh down |
| C7 | Combined benchmark |

## M0: Environment bootstrap (user actions, no product code)

**Deliverables:**
- uv installed.
- Python 3.13 pinned.
- Docker Desktop running.
- `HF_HOME` placed on X: or E:.
- `.env` created from `.env.example`.

**Acceptance:**
- `uv --version` works.
- `uv run python -V` reports 3.13.x.
- `docker info` shows a Server.

**Blockers, to be resolved by the user:**
- Start Docker Desktop.
- Approve the uv install: `py -m pip install --user uv` or the standalone installer.
- Approve the downloads: CPython 3.13 (~30 MB) and CUDA torch (≥ 2 GB, EST).

## M1: Skeleton, workspaces, ingestion with provenance

**Deliverables:**
- `backend/` package layout and boundaries (ADR-0001), plus tooling: ruff, mypy, pytest and import-linter (ADR-0002).
- Settings, JSON logging and request ids.
- Alembic migrations for workspace, document, document_version, temporal_fact, document_relation, chunk, ingestion_job and llm_calls.
- Workspace CRUD API.
- Content-addressed blob store.
- Parsers: pdfplumber+pypdf, TXT/MD, and URL via trafilatura with an SSRF guard.
- Dedupe by `raw_sha256` and `text_sha256`.
- Structure-aware chunker with char and page offsets (ADR-0003).
- Ingestion status and errors.
- Postgres job queue and worker.
- Compose file with postgres, qdrant, api and worker (ADR-0014).
- CI workflow.
- arXiv manifest builder: metadata only, rate-limited (ADR-0013).

**Acceptance:**
- **D1:** API tests create, list and delete workspaces, and deletion cascades to rows, points and blobs.
- **D2:**
  - Golden PDF tests: `text[char_start:char_end] == chunk.text` for every chunk, and the page is correct for known sentences.
  - Re-ingesting the same file gives no new version and returns `duplicate`.
  - The SSRF test suite passes.
- **D9 (partial):** arXiv version dates are stored as `temporal_fact` rows with their source, and `ingested_at` stays distinct.
- **D12 (partial):** `docker compose up` passes the health check.

**Blockers:** Docker running. No API keys are needed.

## M2: BM25, dense, hybrid, and an evaluation harness on a public benchmark

**Deliverables:**
- `Embedder` and `Reranker` protocols, plus a sentence-transformers implementation pinned by HF revision.
- Qdrant adapter with mandatory workspace filter (ADR-0005).
- bm25s lexical index per workspace, with an atomic version swap (ADR-0004).
- RRF with k=60.
- Dedupe and metadata filters.
- The `dense`, `bm25` and `hybrid_rrf` configurations.
- Retrieval metrics, cross-checked against ranx.
- BEIR loader for SciFact and SCIDOCS.
- `mezvuat-exp` CLI with MLflow logging (the `ml` compose profile).
- Embedder sweep across the ADR-0006 shortlist.
- Spike: Qdrant v1.20 native BM25 vs bm25s.

**Acceptance:**
- **D3:**
  - Hybrid retrieval returns ranked chunks with doc, page and section references through the API.
  - The isolation test passes across 2 workspaces.
- **D6:**
  - Re-running one experiment YAML reproduces the metrics; differences from GPU nondeterminism are reported.
  - MLflow runs record git SHA, dataset hash, model revisions and seed.
  - BM25 nDCG@10 on SciFact lands within ±0.02 of a published BM25 reference, with the source cited in the report.
  - Measured latency and indexing-cost table for this hardware.

**Blockers:**
- Approval to download the BEIR datasets (~0.1 GB, EST) and the models (~0.1–1.5 GB, EST).
- GPU CUDA check. If CUDA fails, everything still works on CPU, just more slowly.

## M3: Reranking and the domain dataset

**Deliverables:**
- CrossEncoder reranker and the `hybrid_rrf_rerank` configuration, with a per-request toggle.
- Rerank-depth sweep.
- Demo corpus download: CC-licensed arXiv papers, 100–150 papers, after the user approves the size.
- Doc-family splits.
- Synthetic query generation with lexical-overlap controls, and cross-split dedupe.
- Pooling tool and a minimal labelling UI or CLI.
- Labelling guideline.
- Human labels for val and test, LLM-silver labels for train (ADR-0007).

**Acceptance:**
- **D4:** reranking can be switched on and off per request and per experiment. The four configurations are compared on SciFact and the domain val split, with CIs.
- **D5:**
  - The domain dataset is versioned with a hash.
  - test ≥ 150 and val ≥ 100 human-labelled queries.
  - Inter-annotator κ reported.
  - judged@10 coverage reported per system.
- The split leakage tests pass.

**Blockers:**
- **LLM key** (`GROQ_API_KEY`) for query generation and silver labels. As an alternative, ModelMesh, if it is already up.
- **Human labelling time**, about 3,000 judgments (EST). The product owner decides who labels.
- Corpus download approval.

## M4: LangGraph workflow and citations (direct provider)

**Deliverables:**
- `LLMClient` with `DirectBackend` (Groq), schema validation and repair (ADR-0011).
- LangGraph graph: 8 stages, state schema, bounds and Postgres checkpointer (ADR-0008).
- Evidence registry with deterministic quote and span checks (ADR-0009, part 1).
- `/runs` API with SSE progress and answer streaming.
- `llm_calls` recording.

**Acceptance:**
- **D7:** graph tests with a fake LLM cover every edge, `MAX_ITER`, budget exhaustion and retrieval failure. `recursion_limit` is set explicitly.
- **D8 (partial):** every citation in the final answer resolves to a registry id whose quote is a substring of the chunk, mapped to page and offsets. A fabricated id is rejected.
- **D11 (direct half):** a full run works with only `GROQ_API_KEY`.

**Blockers:**
- `GROQ_API_KEY`. Free-tier rate limits are expected to throttle testing.
- **Product-owner data-policy decision** (ADR-0011, options A–E).
  - Under the default `no_training_providers` policy, Groq is `unknown` and therefore ineligible.
  - Until the PO decides, M4 is built and tested against a fake `LLMClient`.
  - Real calls start only with an approved option, for example E (public demo corpus) or A (approved Groq terms).

## M5: Verification and temporal reasoning

**Deliverables:**
- NLI verifier with threshold calibration on a small hand-labelled set.
- LLM-judge fallback with quote re-check.
- Conflicting-evidence section.
- Injection flagging and the adversarial test set.
- As-of and latest-version resolution, with unknown-date handling and date display in answers (ADR-0010).
- Temporal question set (≥ 30).

**Acceptance:**
- **D8:** claims with failing quotes are dropped and listed as limitations. Adversarial suite: 0 unknown citations reach the user, and no system-prompt leakage.
- **D9:**
  - `ingested_at` never decides "latest".
  - Same-year, year-precision dates do not resolve an as-of query.
  - Answers show dates by kind.

**Blockers:** NLI model download approval. LLM key.

## M6: Evaluation (RAGAS and custom metrics), Langfuse and MLflow

**Deliverables:**
- `eval` dependency group: RAGAS 0.4 collections metrics.
  - The judge is pinned to an explicit model, with `task_type=evaluation_judge` and `routing.fallback: "none"`.
  - The judge's model family differs from the generator's.
  - ModelMesh extensions reach the judge through an httpx transport hook (ADR-0012).
- Custom metrics: citation validity, support, precision and recall; temporal accuracy; abstention; latency; cost.
- Batch RAG eval in MLflow, linked to the Langfuse trace ids.
- Langfuse SDK v5 callback handler with content masking, as a no-op when not configured.
- User feedback endpoint feeding Langfuse scores.

**Acceptance:**
- **D10:**
  - One command runs answer-level evaluation over the 4 retrieval configurations and produces an MLflow-backed report.
  - `citation_validity` = 1.0.
  - When the evaluation or Langfuse is down, user runs still succeed (test).
- **C3 (Mezvuat side):** every `llm_calls` row carries `run_id`, the client request id and the Langfuse trace id.

**Blockers:**
- Judge LLM key and quota.
- Product-owner decision on Langfuse Cloud vs self-host vs none, and on whether masked traces may leave the machine.

## M7: Frontend

**Deliverables:** Next.js 16 app with:
- workspaces;
- document list showing metadata, versions, dates by kind and ingestion status;
- upload and URL ingest;
- research chat with streamed progress per node;
- evidence viewer that highlights the exact span and page and shows injection-suspect warnings;
- conflict and limitations panels;
- fallback and backend badges;
- feedback buttons.

API types come from the generated OpenAPI. Markdown is rendered sanitized.

**Acceptance:**
- **D1, D2, D8 in the UI:** clicking a citation opens the right document, page and highlighted span.
- **D12:** the frontend runs in compose.
- Playwright smoke test (dev dependency, justified at M7).

**Blockers:** none beyond the earlier milestones.

## M8: ModelMesh integration and the combined benchmark

**Preconditions:**
- ModelMesh contract frozen at 1.0. The current baseline is 1.0-rc2, with all CRs accepted by Mezvuat (CQ-1 is a clarification only).
- A ModelMesh instance and API key.
- The data-policy decision (ADR-0011).

**Deliverables:**
- `ModelMeshBackend` per rc2/1.0, generated from `openapi-v1.yaml`:
  - Start-up `GET /v1/contract` negotiation, sending only the listed extensions, with hard failure if `cache.namespace` or `routing.data_policy` is unsupported.
  - `GET /v1/routing/calibration`, so `min_quality` is sent only when `calibrated: true`.
  - Flat metadata, with `correlation_id` matching `[A-Za-z0-9_:.-]{1,128}`.
  - `stream_options.include_usage: true`.
- Fallback rules per contract §3.1:
  - Network errors and 500/502/503/504 trigger fallback.
  - **429 means back off for `Retry-After`, never fallback.**
  - Pre-commit stream failures are handled as HTTP errors.
  - `stream_interrupted` and a missing `[DONE]` mean the partial output is discarded and the call is retried once.
- Client circuit breaker.
- `context_fingerprint` (`sha256:<64 lowercase hex>`), `cache.namespace` per workspace, and cache invalidation on workspace delete.
- `traceparent` propagation, with `modelmesh.trace_id` stored.
- `/v1/feedback/batch` poster (`metric_version`, `attribution`, `eval_run_id`), plus `DELETE /v1/feedback/{id}` on data deletion.
- Contract tests generated from the OpenAPI spec.
- Combined benchmark script: the same eval set run through `direct` and through ModelMesh under 2–3 routing policies, reporting quality from Mezvuat's metrics and cost and latency from `GET /v1/requests/{id}`.

**Acceptance:**
- **C1, C5:** a full research workflow runs through ModelMesh via its public API only.
- **C6:** with ModelMesh stopped mid-session, runs complete via Groq with fallback badges (if the data policy allows). Tests show no fallback on 4xx, including 429.
- **C3, C4:** run ↔ mmr ids ↔ feedback labels can be joined, and feedback is accepted idempotently.
- **C2:** both stacks start and stop independently.
- **C7:** the combined benchmark report is generated from data. No hand-entered numbers.
- **D11:** done.

**Blockers:**
- ModelMesh availability and a key.
- The contract freeze.

## M9: Reranker fine-tuning experiment

**Deliverables:**
- Hard-negative mining restricted to train families.
- `CrossEncoderTrainer` runs: MiniLM-L6 with 3 seeds and an LR grid, and optionally gte-modernbert-base.
- Validation-based selection.
- Single test evaluation with paired bootstrap CIs.
- Guardrails on SciFact and SCIDOCS.
- Generated report (ADR-0007 §F).

**Acceptance:**
- **D4 and D6:** the decision rule is applied as pre-registered, and the outcome is reported either way, including "fine-tuning does not win".
- Training asserts no test-family data was used.

**Blockers:**
- A working GPU (CPU training of MiniLM is possible but slow).
- The M3 labels.
- Thermal throttling: runs must checkpoint and resume.

**Note:** M9 depends only on M3, not on M4–M8. It may run in parallel with M4–M7 once M3's labels exist.

## Cross-cutting rules (every milestone)

- Each milestone ends with an executed verification report: commands plus outputs. Nothing is claimed as done without one.
- No benchmark number appears in docs unless an MLflow run produced it.
- New dependencies are justified in a PR or ADR note.
- Secrets live only in `.env`.
- Anything blocked is isolated behind its interface, with a reproducible alternative, such as a fake LLM in tests. Mock paths are never used as production paths.
