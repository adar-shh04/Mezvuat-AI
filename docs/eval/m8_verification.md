# M8 (Mezvuat side) and M6 scaffolding: verification report

Executed 2026-11-17 on the dev machine (Windows 11, Python 3.13, Node 24), branch `phase-2`, from Git Bash after
`. scripts/dev-env.sh`.

**Nothing here talked to a real ModelMesh, Groq, Ollama or Langfuse.** ModelMesh was busy with its own benchmark on
port 8100, and Mezvuat was not allowed to send it anything. Every ModelMesh behaviour is tested against a mocked
ModelMesh (respx) built from the frozen contract (`docs/contract/openapi-v1.yaml`). The mock serves the spec's own
examples, and both the requests the client sends and the mocked responses are validated against the spec's schemas
with `jsonschema`. No benchmark or evaluation has run, and no numbers are reported.

## Quality gates

| Gate | Command (from `backend/` unless noted) | Result |
|---|---|---|
| Lint | `uv run --no-sync ruff check .` | All checks passed |
| Format | `uv run --no-sync ruff format --check .` | 158 files already formatted |
| Types | `uv run --no-sync mypy` | no issues in 107 source files |
| Import contracts | `uv run --no-sync lint-imports` | 7 kept, 0 broken |
| OpenAPI export | `uv run --no-sync python ../scripts/export_openapi.py --check` | up to date |
| Licences | `uv run --no-sync python ../scripts/check_licenses.py` | 221 packages, 0 need attention |
| Tests, database up | `docker compose -f ../infra/compose.yaml up -d postgres qdrant`, then `uv run --no-sync pytest -q` | **436 passed** (339 before this work) |
| Frontend (from `frontend/`) | `npm run typecheck`, `npm run lint`, `npm run build`, `npx playwright test` (dev server and `PW_SERVER=start`) | pass; Playwright 4 passed in both modes |

The Postgres and Qdrant containers were stopped straight after each integration run.

## M8 acceptance: what the mocked tests cover, and what needs live integration

| Item | Mocked tests | Needs live ModelMesh |
|---|---|---|
| `GET /v1/contract` negotiation; only listed extensions sent; hard failure without `routing.data_policy` or `cache.namespace` | `test_modelmesh_client.py` (negotiation section) | that the real server's contract lists them |
| `min_quality` only when calibrated | calibrated, uncalibrated, calibration down | real calibration report |
| Flat metadata, valid `correlation_id`; `include_usage`; fingerprint and namespace formats | format section, contract tests | none |
| Fallback per §3.1, parametrised over every error status the spec lists; 429 backs off and never falls back; pre-commit stream 5xx falls back; `stream_interrupted` and missing `[DONE]` raise for the discard-and-retry-once in the respond node | fallback and streaming sections | real error bodies and timings |
| Circuit breaker with the `/v1/health` probe | breaker section | none |
| D1 on fallback: private never reaches Groq; no local model means a clear failure | D1 section, `test_modelmesh_workflow.py` | Ollama behind fallback |
| **C1, C5**: a full research run through ModelMesh's public API | `test_full_research_run_through_modelmesh` (mocked) | **yes**: a real run |
| **C6**: ModelMesh stopped mid-run, Groq fallback, badges; no fallback on 4xx including 429 | `test_modelmesh_stopping_mid_run_...`, fallback table, Playwright fallback test | **yes**: stopping the real stack |
| **C3, C4**: run ↔ X-Request-ID ↔ mmr id ↔ trace id ↔ feedback; idempotent feedback | workflow test, `test_modelmesh_outbox.py` (Postgres), feedback contract tests (replayed, conflict) | **yes**: the real join, and whether ModelMesh's `trace_id` equals the Langfuse trace id |
| Cache purge and feedback deletion on workspace delete | outbox integration tests | real purge counts |
| **C2**: stacks start and stop independently | not tested here | **yes** |
| **C7**: combined benchmark from data | plumbing with fakes (`test_combined_benchmark.py`) | **yes**: `docs/eval/combined_benchmark/report.md` says not yet run |

## M6 scaffolding

Built and tested with fakes only: custom metrics, RAGAS 0.4.4 behind a pluggable pinned judge, the MLflow-backed
`answer-eval` runner, gold-item export from the temporal review (read-only), ModelMesh labels from evaluation results,
and masked Langfuse tracing (the real SDK with an in-memory exporter: no question, document or system-prompt text
reaches it; runs complete when export fails). Commands: `docs/eval/answer_eval/README.md`.

## Not verified

- Anything against a real ModelMesh, Groq, Ollama, Langfuse Cloud or a real judge.
- That RAGAS behaves the same with a real judge as with the scripted one, and how many judge calls a run costs.
- MLflow logging against the real SQLite store: the runner was tested with a fake tracker.
- The Docker image with `langfuse` now a runtime dependency (not built: memory).
- Before RAGAS analytics were switched off (`RAGAS_DO_NOT_TRACK=true`), one test run of the RAGAS tests may have sent
  RAGAS's anonymous usage event (metric names, no data) to its vendor. Analytics are off in the code and tests now.
