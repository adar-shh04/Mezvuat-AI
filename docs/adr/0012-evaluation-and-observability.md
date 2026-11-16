# 0012. Evaluation and observability

Proposed · 2026-11-08 · Spec §1.3G, §3.4, §4.3 ("evaluation failure must not fail the workflow"), §8 item 8

## Context

The spec asks for RAGAS plus custom metrics, and splits the tools: Langfuse for LLM and workflow tracing, MLflow for experiments, with no duplication between them. Application quality results must be associated with ModelMesh executions.

## Options considered

- **RAGAS 0.4.3** (Apache-2.0, released 2026-01-13).
  - Current API: `ragas.metrics.collections` with `await metric.ascore(...)`. `evaluate()` is deprecated, and so are the legacy `SingleTurnSample` metrics, which go away in 1.0.
  - Heavy dependencies: langchain, langchain-community, openai, instructor, datasets, scikit-network.
  - We can isolate it in an `eval` dependency group, or reimplement the metrics ourselves. Isolating is chosen: the spec names RAGAS, and its metrics are well understood.
- **Tracing:**
  - Langfuse alone is chosen.
  - MLflow 3 GenAI tracing would duplicate Langfuse. Rejected.
  - Bare OpenTelemetry into Jaeger: no LLM-aware UI. Rejected.
- **Experiments:** MLflow 3.18 (Apache-2.0) is chosen. Langfuse "datasets/experiments" would duplicate MLflow for retrieval work. Rejected.

## Decision

### Responsibilities

| Concern | Tool | Not in |
|---|---|---|
| Per-run traces: graph node spans, LLM generations (prompt and response, token usage, latency, backend, mmr id), retrieval spans (config, k, latency, ids only) | **Langfuse** | MLflow |
| Online scores attached to a trace: user feedback, cheap deterministic metrics (citation_validity, fallback_used) | **Langfuse** scores | MLflow |
| Offline experiments: retrieval configs, reranker training, batch RAG evaluation (RAGAS + custom), dataset versions, model artifacts | **MLflow** | Langfuse |
| Operational logs and metrics | JSON logs, with request and run ids | |

**Rule:** anything comparing configurations or models across a dataset belongs in MLflow. Anything about one user request belongs in Langfuse. A batch eval run stores the list of Langfuse trace ids it produced as an MLflow artifact, which links the two **without copying data**.

### RAG answer evaluation (offline job, `eval` group, M6)

**Dataset:**
- Built from the domain **test** questions (ADR-0007) plus the temporal set (ADR-0010).
- Each item has a reference answer and gold evidence chunk ids, written by a human for test.
- Running the 4 configurations end to end, each through the full workflow, gives answer-level comparisons.

**RAGAS metrics (0.4 collections API):**
- `ContextPrecision` (with reference)
- `ContextRecall` (needs reference)
- `Faithfulness`
- `AnswerRelevancy` (needs an LLM **and** embeddings)

**Judge LLM** (contract 1.0-rc2 §2.2):
- Configured separately from the generator through the `EVAL_JUDGE_*` env vars.
- Every judge call uses `task_type: "evaluation_judge"` and a **pinned explicit model** with `routing.fallback: "none"`, so the evaluator stays fixed even when the router changes.
- Through ModelMesh, judge requests are excluded from router quality-training data.
- If the pinned judge is unavailable, the evaluation run **fails**. It never switches judges silently.
- The judge model id and version are logged. The judge must come from a **different model family** from the generator under test; if that cannot be arranged, it is reported as a limitation.

**Wiring the judge into RAGAS:**
- Pass `llm_factory(model, client=AsyncOpenAI(base_url=…, http_client=…))`, an OpenAI-compatible client pointed at ModelMesh or Groq.
- RAGAS does not send ModelMesh extensions itself. A custom httpx transport on that client injects:
  - flat `metadata`: `application`, `task_type=evaluation_judge`, `correlation_id=eval_<run>`
  - `routing.fallback: "none"`
  - `cache.namespace`
  - `routing.data_policy`
- Only fields listed by `GET /v1/contract` are injected.
- Compatibility of this wiring with `llm_factory` is UNVERIFIED and is spike-tested in M6.
- Judge calls carry evidence, so the workspace data policy applies to them too (ADR-0011, open PO decision).
- Embeddings for AnswerRelevancy: our local bge-small, through RAGAS's embedding interface. That RAGAS accepts a custom embedding class is checked in M6, with a thin adapter as the fallback.

**Custom metrics (Mezvuat-owned, deterministic where possible):**

| Metric | Definition |
|---|---|
| `citation_validity` | Share of citation markers that resolve to registry ids whose quotes pass the span check. Must be 1.0. |
| `citation_support_rate` | Share of cited claims with NLI-entailed or judge-verified support |
| `citation_precision` / `citation_recall` | Cited chunks vs. gold evidence chunks |
| `unsupported_claim_rate`, `claims_dropped` | From the `verify` output |
| `temporal_accuracy` | ADR-0010 temporal set: correct version cited, or correct abstention |
| `abstention_correctness` | Questions with no answer in the corpus (≥ 20 items): the system must say so |
| `latency_ms` | Total workflow time and per node: p50/p95 |
| Cost | LLM calls and tokens per run; `estimated_usd` taken from ModelMesh, or computed from the provider price list and labelled as an estimate |

**Robustness:** evaluation failures never affect user requests (§4.3). Online scoring runs as a fire-and-forget worker job, and errors are only logged.

### Correlation (§8 item 8)

**Join keys** live in Postgres table `llm_calls`, which is the system of record:

| Key | Meaning |
|---|---|
| `run_id` | Mezvuat research run = ModelMesh `correlation_id` = Langfuse trace metadata `run_id` and session |
| `langfuse_trace_id` | |
| `client_request_id` | `X-Request-ID` |
| `modelmesh_request_id` | `mmr_…`, present on errors as well (rc2) |
| `modelmesh_trace_id` | W3C trace id from `modelmesh.trace_id`. ModelMesh spans are children of Mezvuat's span. |
| `traceparent` | |
| `task_type`, `workflow_node` | |

**Flow:**
1. The eval job writes per-run and per-call scores to `eval_results`, joined via `run_id` and `llm_calls`.
2. The job pushes aggregates to MLflow.
3. It pushes per-request labels to ModelMesh `POST /v1/feedback/batch`, following ADR-0011:
   - always with `metric_version` and `eval_run_id = "mlflow:<run_id>"`;
   - `attribution: direct` for labels on the producing request;
   - `attribution: workflow` for end-to-end scores.
4. The combined benchmark (M8) joins Mezvuat quality with ModelMesh execution data **through the public API only**:
   - `GET /v1/requests?correlation_id=<run_id>` (or `/v1/requests/{mmr_id}`) supplies cost, latency and routing.
   - Calibration status from `GET /v1/routing/calibration` is recorded alongside the results.

**Langfuse SDK v5.x** (MIT, OpenTelemetry-based):
- `langfuse.langchain.CallbackHandler` on LangGraph `invoke`.
- Plus manual spans for retrieval and verification.
- **Prompt and document content in traces is masked by default**, through a mask function that keeps ids and lengths. Full content needs `LANGFUSE_CAPTURE_CONTENT=true`. This setting matters most with Langfuse Cloud (ADR-0014).
- If Langfuse keys are absent, tracing is a no-op, and the app runs normally.

**MLflow:** tracking server in compose (ADR-0014). Postgres backend store (the same Postgres instance, a separate `mlflow` database) and a local artifact volume.

## Amendment, 2026-11-17: M6 scaffolding as built (tested with fakes only; no evaluation has run)

The owner's labels are not finished, so nothing here has produced a number yet. The commands are in
`docs/eval/answer_eval/README.md`.

- **Pluggable judge** (`evaluation/judge.py`): anything with `model` and `structured(prompt, response_model)`.
  Production judges use Mezvuat's own `LLMClient` with `task_type: evaluation_judge`, a pinned `model`, temperature 0
  and seed 0, so ModelMesh gets `routing.fallback: "none"`, `cache.mode: "exact"` and no `min_quality` from the same
  code as every other call. **This replaces the planned httpx transport hook on RAGAS's OpenAI client**: RAGAS calls
  the judge through a thin `InstructorBaseRagasLLM` adapter instead, so no OpenAI client is involved. The direct judge
  is Groq with the model pinned. Judge calls always send `data_policy: "any"`, so the runner refuses any workspace
  that is not a verified public-corpus workspace (D1).
- **RAGAS 0.4.4** (`evaluation/ragas_metrics.py`, `eval` group): `Faithfulness`, `AnswerRelevancy` (Mezvuat's own
  embedder through a `BaseRagasEmbedding` adapter), `ContextPrecisionWithReference`, `ContextRecall`. Missing inputs
  (no reference, no contexts, an abstained answer) skip a metric; NaN is "not scored". RAGAS's output models were
  checked to fit the portable schema subset, so strict JSON schema works for the judge. RAGAS analytics are turned
  off (`RAGAS_DO_NOT_TRACK=true`) before RAGAS is imported.
- **Custom metrics** (`evaluation/answer_metrics.py`): `citation_validity`, `citation_support_rate`,
  `citation_precision`/`citation_recall` (against `gold_chunk_ids`), `unsupported_claim_rate`, `claims_dropped`,
  `abstention_correct`, `temporal_accuracy`, and per-run calls, tokens, LLM latency and ModelMesh's estimated cost
  (absent on the direct path, never estimated by Mezvuat). A metric that does not apply to an item is absent, not 0.
- **Batch runner** (`evaluation/answer_eval.py`, `mezvuat-exp answer-eval`): every gold item through the full
  workflow per retrieval configuration (`bm25`, `dense`, `hybrid_rrf`, `hybrid_rrf_rerank`); an MLflow parent run with
  one nested run per configuration; per-item rows and the Langfuse trace ids as MLflow artifacts; `results.json` and
  `report.md` generated from the same rows. `--post-feedback` queues ModelMesh labels through the M8 outbox:
  `faithfulness` on the synthesis request and `answer_relevancy` on the final answer (`attribution: direct`), and
  `task_success` on every request of the run (`attribution: workflow`). `eval_results` is not a table: the rows live
  in `results.json` and in MLflow, joined to `llm_call` by `run_id`.
- **Gold items**: `mezvuat-exp export-gold` merges the temporal drafts with the owner's latest review decision
  (read-only; rejected and unreviewed items are left out). Domain test questions need reference answers and gold
  chunk ids in the same JSONL format before they can join.
- **MLflow**: local SQLite at `X:/mezvuat-cache/mlflow/mlflow.db` through `MLFLOW_TRACKING_URI` (set by
  `scripts/dev-env.sh`), not the compose tracking server, while memory is tight.
- **Langfuse** (`observability/tracing.py`): `langfuse` moved to the runtime dependencies, because tracing happens in
  the API. With either key unset, or the SDK missing, tracing is a no-op. One trace per run, with the trace id derived
  from the run id (`Langfuse.create_trace_id(seed=run_id)`), stored in every `llm_call.langfuse_trace_id`; the
  LangChain callback handler traces the graph. The `mask` function runs in the process before export and replaces
  every string that is not an identifier with its length and a short hash; `LANGFUSE_CAPTURE_CONTENT=true` turns
  that off. Tests run the real SDK with an in-memory exporter and check that the question, the documents and the
  system prompt never reach it, and that runs still complete when export fails or tracing cannot start. User ratings
  become a `user_thumbs` score on the run's trace. While a run is traced, its OpenTelemetry span is current, so the
  `traceparent` sent to ModelMesh carries the Langfuse trace id; whether ModelMesh's `trace_id` then equals it is
  checked at live integration.

## Consequences

- The eval group is heavy, but it never ships in the API image.
- Running RAGAS 0.4 means Python ≤ 3.13 on Windows, because of scikit-network (ADR-0002).
- LLM-judge metrics cost tokens and hit rate limits, so eval batches must be rate-limited and cached. Judge results are cached by (item, answer hash, judge model).

## Verification plan

- **Unit tests** for each custom metric on fixtures.
- **RAGAS smoke test** on 3 fixture items with a scripted fake judge: the API wiring works without network access.
- **Correlation test:** after one run, every `llm_calls` row has `run_id`, `client_request_id` and `langfuse_trace_id`, plus `modelmesh_request_id` when the backend is ModelMesh. The Langfuse trace shows the same ids.
- **Isolation test:** with Langfuse unreachable, a research run still succeeds, and a warning is logged.

## Sources (accessed 2026-11-08)

- pypi.org/project/ragas (0.4.3)
- docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
- docs.ragas.io metric pages for answer_relevance, context_precision, context_recall and faithfulness
- docs.ragas.io/en/stable/howtos/customizations/customize_models/
- pypi.org/pypi/langfuse (5.2.0)
- langfuse.com/integrations/frameworks/langchain
- pypi.org/pypi/mlflow (3.18.0)
- mlflow.org/docs/latest/self-hosting/
