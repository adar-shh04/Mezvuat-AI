# 0011. Inference layer: provider-independent LLM interface, ModelMesh client, direct-provider fallback

Proposed · 2026-11-08 · **Amended 2026-11-08 for contract 1.0 and decision D1 (`docs/DECISIONS.md`)** · Spec §1.5 Inference, §3, §4.1, §4.3, §5.3, §8 items 7 and 8 · Contract: `docs/contract/inference-api-v1.md` (1.0, frozen), `docs/contract/openapi-v1.yaml` (normative), `docs/contract/CR_RESOLUTIONS.md`

## Context

Mezvuat must work **without ModelMesh**. When ModelMesh cannot serve a request, a configured direct provider takes over (spec §4.3). The integration may use only the public contract.

There are two kinds of LLM call:
- **Structured calls.** Most nodes (ADR-0008) need JSON-schema output.
- **One streamed call:** `final_response`.

No API keys exist yet.

## Options considered

**Client implementation:**

| Option | Assessment |
|---|---|
| (a) OpenAI Python SDK pointed at ModelMesh/Groq | Familiar, but extensions go through `extra_body`, and the error envelope needs custom parsing anyway. |
| (b) **httpx + Pydantic models generated from `openapi-v1.yaml`** | **Chosen.** Explicit, typed, and the same HTTP stack serves both backends. |
| (c) LiteLLM | A second gateway inside Mezvuat that duplicates ModelMesh. Rejected. |

**Direct provider:**

| Provider | Structured output | Data | Verdict |
|---|---|---|---|
| **Groq** | `json_schema` + `strict: true` (constrained decoding) on gpt-oss-20b/120b and qwen3.8-27b. No streaming together with structured output. | No retention of inference data by default; Zero Data Retention available. A no-training statement has not been verified; ModelMesh attests Groq as `data_use: unknown`. | **Chosen for public-corpus workspaces only** (D1, D4). |
| **Ollama** (local) | Structured output depends on the model and is unverified. | Stays on the machine. | **Chosen for private workspaces** (D1). Not installed yet. |
| Gemini | JSON schema supported; streaming | Free tier: prompts **are used to improve Google products**; human review possible (terms effective 2026-03-23). Paid tier: not used. | Free tier unacceptable for user documents (§5.3). Paid tier is one of the PO options. |

## Decision

### Interface (`mezvuat.inference`)

```python
class LLMRequest(BaseModel):
    task_type: TaskType                      # contract §2.2 (rc2 vocabulary, incl. evidence_assessment, evaluation_judge)
    messages: list[ChatMessage]
    response_schema: type[BaseModel] | None  # → response_format json_schema; schema restricted to the portable subset (contract §2.5)
    temperature: float = 0.0
    max_tokens: int
    seed: int | None = None
    stream: bool = False
    model: str | None = None                  # explicit model pin (required for evaluation_judge); None → "auto"
    constraints: RoutingHints                 # policy, max_cost_usd, max_latency_ms, timeout_ms, min_quality?, data_policy …
    cache: CacheHints                         # mode, ttl_s, context_fingerprint, namespace
    correlation: Correlation                  # run_id, workflow_node, end_user_hash, prompt_version, iteration

class LLMClient(Protocol):
    async def complete(self, req: LLMRequest) -> LLMResult: ...
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMStreamEvent]: ...
```

`LLMResult` fields:
- parsed output and raw text
- `backend` (`modelmesh` | `direct`)
- `model`
- `client_request_id`
- `modelmesh_request_id` and `trace_id`, when present
- usage, with `source`
- `cost_estimated_usd`
- latency
- cache status
- `structured_output.mode`
- fallback info

Anything the direct backend cannot report is `None`. It is never silently estimated.

### Start-up negotiation (contract §0, §2.4, §4)

The `ContractCapabilities` service is cached in the process.

1. **At start-up, call `GET /v1/contract`** (unauthenticated) and cache:
   - `contract_version`
   - `extensions[]`, e.g. `routing.timeout_ms`, `cache.namespace`
   - `task_types[]`, `capabilities[]`, `routing_policies[]`, `feedback_metrics[]`
2. **Refresh** every 10 minutes, and once immediately after any `400 invalid_request` with code `unknown_parameter`.
3. **Request builder.** It serialises **only** extension fields whose dotted path appears in `extensions[]`. A dropped field is logged once per process as `extension_unsupported:<path>`. A **hard requirement** is different:
   - Hard requirements are `cache.namespace` whenever a workspace is involved, and `routing.data_policy`.
   - If one is unsupported, the call fails with `inference_contract_mismatch` instead of being sent without it. Sending without it would weaken isolation or privacy silently.
4. **`task_type` fallback.** If a `task_type` is not in `task_types[]`, send `general` plus a free-form metadata key `task_type_intended`.
5. **Calibration.** Call `GET /v1/routing/calibration` at start-up and refresh hourly. `min_quality` is sent **only** for task types where `calibrated: true`, and only if a value is configured for that task type. rc2 reports `calibrated: false` everywhere until ModelMesh M6, so in practice nothing is sent. `evaluation_judge` is never calibrated in v1, so it never gets `min_quality`.
6. **ModelMesh unreachable at start-up.** Mark the backend unavailable, so calls use the direct path under the fallback rules, and retry the probe in the background.

### Backends

#### `ModelMeshBackend`

**Endpoint:** `POST {MODELMESH_BASE_URL}/v1/chat/completions`.

**Model and routing:**
- `model="auto"`, unless the request pins a model.
- Default `routing.policy` per task type: `cost_optimized` for `query_classification` and `query_rewriting`, `quality_balanced` otherwise.
  - The earlier `quality_balanced`-only workaround (CQ-1) is no longer needed. Contract 1.0 (F1) makes the `cost_optimized` default floor of 0.6 apply only to calibrated task types and never hard, and F2 defines `quality_balanced` and `quality_max` when predicted quality is unavailable (rules policy).
  - Warnings such as `default_min_quality_skipped:uncalibrated` are logged, not treated as errors.
- `routing.timeout_ms` = the node's remaining deadline. The httpx client timeout is set to that value + 2 s.
- `routing.fallback` is left at its default (`within_constraints` for `auto`), except for the judge (below).

**Metadata** is **flat**, an OpenAI-shaped string→string map of at most 16 keys with values of at most 512 characters.

| Key | Value |
|---|---|
| `application` | `"mezvuat"` |
| `task_type` | the request's task type |
| `correlation_id` | the research `run_id`, a ULID rendered `run_<ULID>`; matches `^[A-Za-z0-9_:.-]{1,128}$` and is validated client-side |
| `end_user_id` | `"hash:" + hex(HMAC-SHA256(server secret, user_id))[:64]` (≤ 128 chars) |
| `workflow_node`, `prompt_version`, `iteration` | free-form keys, all strings |

There is no nested `tags` object, and no `task_type_intended` key except in the fallback case above.

**Structured calls:**
- Send `response_format: {type: "json_schema", strict: true, schema}` and `required_capabilities: ["json_schema"]` (native modes only).
- `citation_verification` and `evidence_synthesis` send `["json_schema_strict"]`, because citation correctness depends on schema-valid output and constrained decoding is the strongest guarantee. If this yields `422 no_eligible_model`, the node retries **once** with `["json_schema"]`, and `structured_output.mode` is logged.
- **Schemas** are generated from Pydantic and checked in CI against the **portable subset** (contract §2.5):
  - Every property is required.
  - `additionalProperties: false`.
  - Nullable is written as `["T","null"]`.
  - No `$ref`, `oneOf`, `anyOf`, `pattern`, `minLength` or `maxLength`.
  - Nesting depth ≤ 5.
- Length limits are enforced in Pydantic after parsing, not in the schema. One example is the 300-character quote cap in ADR-0009.

**Streaming (`final_response`):**
- Send `stream: true` and `stream_options: {include_usage: true}`.
- The last chunk before `[DONE]` has `choices: []` and carries `usage` plus `modelmesh`. The client must not index `choices[0]` on it.
- Ignore `: keep-alive` comment lines.

#### `DirectBackend` (Groq)

- **API:** the OpenAI-compatible endpoint, with one fixed model per task type set in config.
- **No intelligent routing.** Mezvuat does not route here; that would duplicate ModelMesh (§4.2).
- **Structured output:** `strict: true` with the same portable-subset schemas.
- **Streaming:** `stream_options.include_usage: true`.
- **Data policy:** the backend **enforces the workspace data policy itself** (contract §3.1): Groq serves public workspaces only.

#### Both backends

- Output is validated against the Pydantic model.
- On failure, there is one repair retry with the validation error fed back.
- After that the call raises `structured_output_invalid` and the node fails gracefully.
- On ModelMesh, a 200 response is already schema-valid (contract §2.5). Mezvuat still validates, because Pydantic carries extra constraints.

### Mode and fallback rules (adopts contract §3.1; no CR-15)

`INFERENCE_MODE` takes one of three values:
- `direct`
- `modelmesh` (no fallback)
- `modelmesh_with_fallback`

**429 is not a fallback trigger.** Mezvuat adopts ModelMesh's rule. A key-level 429 means Mezvuat exceeded limits that the operator set on purpose, while ModelMesh itself is healthy. Upstream exhaustion comes back as 503 `upstream_capacity`, which *is* a trigger.

No Mezvuat requirement is broken by this:
- Spec §4.3 asks for fallback when ModelMesh is **unavailable**, and it is not unavailable here.
- §5.3 favours honouring configured limits.
- On a 429 the user sees a bounded wait or an explicit "rate-limited, retry" error, never a fabricated answer.

| ModelMesh outcome | Mezvuat action |
|---|---|
| Connect, DNS or TLS error; read timeout before response headers | Retry once with jitter, then **fallback** |
| **500** `internal_error`, **502** `upstream_error` (including `schema_validation_failed`), **503** `service_unavailable` (including `upstream_capacity`), **504** `upstream_timeout` | Retry with backoff: base 0.5 s, factor 2, full jitter, at most 2 retries, bounded by the node deadline. If 503 has `Retry-After`, honour it when it fits. Then **fallback**. |
| **429** `rate_limited` | Wait `Retry-After` seconds (always present) if that fits in the node's remaining deadline, then retry, at most twice. Otherwise **fail the step** with `inference_rate_limited`. The workflow reaches `respond`/`fail` with an explicit limitation. **No direct fallback.** |
| 400, 401, 402, 403, 404, 409, 413, 422, 501 | **No fallback.** Surface a typed error. These are configuration, policy or budget decisions; bypassing them would silently violate constraints. |
| **Streaming, failure before the commit point** | It arrives as an ordinary **HTTP error with a JSON body** (rc2 R3), so it is handled exactly like a non-streaming error: 5xx → retry, then fallback; 4xx → no fallback. Nothing has been shown to the user yet. |
| **Streaming, in-stream `stream_interrupted`** (error event, the stream closes **without `[DONE]`**) | Discard the partial output: the UI replaces the partial text with "response interrupted, retrying". Retry the whole `respond` call **once** with a new `X-Request-ID` on the same backend if the deadline allows. Otherwise fail with an explicit error. No mid-answer switch of backend. **A missing `[DONE]` is always treated as incomplete**, even without an error event. |

**`finish_reason`:** values are `stop`, `length`, `tool_calls` and `content_filter`.
- `length` is treated as truncated: the answer is flagged and verification still runs.
- `content_filter` gives an explicit "provider filtered the response" message.

**Client-side circuit breaker:**
- Five consecutive fallback-eligible failures open the breaker for 30 s.
- While it is open, calls go straight to the direct backend.
- The half-open probe is `GET /v1/health`: 200 means `ok` or `degraded`; 503 means `down`.

**Data policy on fallback.** The direct backend applies the same workspace data policy as ModelMesh. If the policy forbids the direct provider, nothing is sent, and the error is `inference_not_permitted`.

**Recording.** Every fallback is written to `llm_calls` with its reason, and the answer shows a fallback badge.

### Evaluation judge calls (ADR-0012)

- `task_type: "evaluation_judge"`.
- **Pinned explicit `model`**, e.g. a `groq/openai/gpt-oss-*` id chosen at M6, from a model family different from the generator under test.
- `routing.fallback: "none"`. This is also the default for explicit models; Mezvuat sets it anyway.
- No `min_quality`.
- `cache.mode: "exact"` and `temperature: 0`, so re-scoring is cheap and reproducible.
- If the pinned model is unavailable, the evaluation run **fails**; it does not silently switch judges.
- Judge calls are not sent to `/v1/feedback`. ModelMesh would mark them `training_eligible: false` anyway.

### Cache, correlation and feedback

**`X-Request-ID`:** a fresh ULID **per HTTP attempt**. It is not an idempotency key; ModelMesh never deduplicates. It is stored in `llm_calls.client_request_id`.

**Stored per call:**
- `X-ModelMesh-Request-ID` (`mmr_…`, always present, also on errors as `error.request_id`)
- `modelmesh.trace_id`
- `client_request_id`
- `correlation_id`

**`traceparent`:** propagated from the current OpenTelemetry span. Langfuse SDK v5 is OpenTelemetry-based. ModelMesh makes its server span a **child** of ours, so both share one W3C trace id. Whether the Langfuse trace id equals the W3C trace id is checked at M6. If it does not, `llm_calls` stores both.

**`cache.namespace`:** `"ws_" + hex(HMAC-SHA256(server secret, workspace_id))[:32]`.
- It matches `^[A-Za-z0-9._:-]{1,128}$`.
- It is sent on **every** call from a workspace, which is the primary cross-workspace isolation within one ModelMesh tenant.
- Workspace deletion calls `POST /v1/cache/invalidate {namespace}`.

**`cache.context_fingerprint`:** `"sha256:" + lowercase_hex(sha256(canonical_json(...)))`, matching `^sha256:[0-9a-f]{64}$`. The input is:

```
{"v": 1,
 "workspace_id": ...,
 "prompt_version": ...,
 "evidence": sorted([[chunk_id, document_version.text_sha256, chunker_version], ...])}
```

- `workspace_id` stays in the hash as defence in depth, independent of the namespace.
- Calls without evidence use `{workspace_id, prompt_version}`.

**Cache modes:**

| Calls | Mode |
|---|---|
| Temperature-0 structured nodes, judge | `exact` |
| `final_response`; any call with temperature > 0 and no seed | `off` (explicit, not left to the rc2 default "bypass") |

`semantic` is not used in v1.

**Feedback** goes to `POST /v1/feedback/batch` (≤ 500 items) from an async job after each evaluation run.

| Field | Value |
|---|---|
| `metric` | the contract vocabulary (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `citation_validity`, `citation_support_rate`, `temporal_accuracy`, `schema_valid`, `task_success`) |
| `metric_version` | always set; part of the idempotency key |
| `eval_run_id` | `"mlflow:<run_id>"` |
| `dataset_version` | set |
| `labelled_at` | set |
| `accepted` | set where Mezvuat has a binary criterion |
| `attribution` | `direct` for labels on the request that produced the evaluated artefact (synthesis → faithfulness; final → answer_relevancy; verification → schema_valid). `workflow` for end-to-end scores such as `task_success`, which are sent to every request in the run and are not training-eligible by contract. |
| `label_source` | `automated_eval`, `human_review` (human-labelled test items) or `user_signal` (thumbs) |
| `notes` | always `null` |

Responses are handled per item:
- `replayed` is fine.
- `conflict` (same key, different score) is logged as an evaluator nondeterminism bug, and **is not overwritten**.
- `not_found` and `invalid` are logged.

Workspace or user deletion calls `DELETE /v1/feedback/{request_id}` for that workspace's stored mmr ids.

### Data policy for LLM calls (decided: D1, options E + C)

The product owner chose public cloud for the public corpus and local models for everything else. `docs/DECISIONS.md` is authoritative.

- **Classification is derived, never declared.** A workspace is `public` only when it has documents and every one is verified against the allowlisted CC-licensed demo corpus by the SHA-256 of its bytes (`workspace.data_classification`, computed by `recompute_classification`). Adding any other document makes it `private`. The API has no way to set it.
- **ModelMesh:** public workspaces send `routing.data_policy: "any"`; private workspaces send `"local_only"`.
- **Direct fallback follows the same rule:**
  - Public workspaces may use Groq (free tier only, D4).
  - Private workspaces use a **direct Ollama backend**, so Mezvuat stays usable without ModelMesh. Its weaker models are accepted.
  - The `DIRECT_PROVIDER_DATA_USE` attestation table from the earlier draft is replaced by this rule.
- **If no permitted backend is available, the call fails** with `inference_not_permitted`. The policy is never widened.
- **Offline jobs obey D1 too:** RAGAS judging and LLM pre-labelling run only on the public corpus.
- **Rate limits.** A new ModelMesh key defaults to 20 RPM, 50K TPM and $0.25/day, and a research run can make 12 calls with large evidence payloads. 429 is handled as specified above; if the defaults block normal use, report it and request explicit limits.

## Amendment, 2026-11-14: direct backends as built in M4

- **Synchronous interface.** `LLMClient.complete()` returns `LLMResult` and `stream()` returns an iterator, for the reason given in ADR-0008's M4 amendment. The fields are unchanged. Each HTTP attempt is an `LLMAttempt` with its own `X-Request-ID`, and every attempt becomes one `llm_call` row, including failed attempts and repair retries.
- **One OpenAI-compatible backend class** serves Groq (`locality: cloud`) and Ollama (`locality: local`, through Ollama's `/v1/chat/completions`). A cloud backend refuses `local_only` requests before sending anything. `DirectClient` picks the public backend for `any` and the local one for `local_only`, and raises `inference_not_permitted` if neither is allowed. It never widens the policy.
- **Configuration.** `OLLAMA_BASE_URL` (unset by default: private workspaces then cannot run), `OLLAMA_MODEL`, `OLLAMA_MODEL_LARGE`, and `DIRECT_PUBLIC_BACKEND` (`groq` by default; `ollama` keeps local development off the Groq quota). `GROQ_MODEL_LARGE` serves `evidence_synthesis`, `final_response` and `citation_verification`; every other task uses `GROQ_MODEL_SMALL`. gpt-oss models are sent `reasoning_effort: "low"`.
- **Retries.** Transport errors and 5xx get at most 2 retries with full-jitter backoff, then `inference_unavailable`. A 429 waits for `Retry-After` when that fits the request deadline, at most twice, then `inference_rate_limited`. Other 4xx responses are `inference_rejected`. Error bodies are reduced to status, type and code, because they can echo prompt text.
- **Ollama structured output** was checked live on Ollama 0.40.16 with `llama3.2:3b`: `response_format: json_schema` is honoured through the OpenAI-compatible endpoint.
- `INFERENCE_MODE=modelmesh*` fails at start-up with `inference_mode_unsupported` until M8.

## Amendment, 2026-11-17: the ModelMesh client as built in M8 (tested against a mocked ModelMesh only)

Code: `backend/src/mezvuat/inference/modelmesh/`. No request has been sent to a real ModelMesh yet; live integration
is a later step.

- **Wire models are hand-written, and checked against the spec instead of generated.** The client parses only the
  fields it uses (`ContractInfo`, `RoutingCalibration`, the `modelmesh` block, `FeedbackRequest`). Contract tests
  load `docs/contract/openapi-v1.yaml` and check, with `jsonschema` (MIT, dev only): every request body the client
  sends, every mocked response, the parse models' required fields against the spec's, and that every extension the
  builder can emit is defined in `RoutingOptions` or `CacheOptions`. The fallback tests are parametrised over every
  error status the spec lists for `POST /v1/chat/completions`. The mocks serve the spec's own examples.
- **Negotiation.** `GET /v1/contract` is read lazily on the first call and cached for 10 minutes; the API also reads
  it at start-up, and a missing `routing.data_policy` or `cache.namespace`, or a major version other than 1, stops
  the API (`inference_contract_mismatch`). An unreachable ModelMesh at start-up is only logged. Calibration is cached
  for an hour; when it cannot be read, nothing counts as calibrated (retried after a minute).
- **`cache.context_fingerprint`** hashes `{v, workspace_id, prompt_version, sorted[(chunk_id, version_id,
  sha256(chunk text))]}`: the SHA-256 of the text actually shown, rather than the version's `text_sha256`, so a
  re-chunked or edited passage also changes the key. Cache `mode` is `exact` for deterministic calls and `off` for
  `final_response` or any sampled call without a seed.
- **`cache.namespace`** uses `MEZVUAT_SECRET_KEY` as the HMAC secret; ModelMesh modes refuse to start without it.
- **`traceparent`.** The active OpenTelemetry span's trace id when there is one (Langfuse, M6); otherwise a trace id
  derived from the run id, so every call of a run shares one trace id. A fresh span id per call.
  `modelmesh.trace_id` is stored in `llm_call.modelmesh_trace_id`.
- **`end_user_id` is not sent**: Mezvuat has no per-user identity yet (one shared API key).
- **Retries.** Network errors: one retry with jitter, then fallback. 500/502/503/504: at most two retries (honouring
  `Retry-After`), then fallback. A 200 with an unreadable body counts as unavailable. A 400 `unknown_parameter`
  re-reads the contract and resends once. A 422 while requiring `json_schema_strict` resends once with `json_schema`.
- **Streams.** The client raises `StreamIncompleteError` on `stream_interrupted` or a missing `[DONE]`; the respond
  node already discards the partial text (`answer_reset`) and retries once with a new `X-Request-ID`. Only failures
  before the commit point fall back; a stream that started on ModelMesh never switches backend.
- **Circuit breaker** counts failed *calls* (after their retries), not HTTP attempts. 4xx and 429 reset the count.
- **Recording.** Every attempt becomes an `llm_call` row with `backend`, the `mmr_` id, the trace id, cache status,
  estimated cost and, on direct attempts made in fallback, `fallback_reason` (`modelmesh_503`,
  `modelmesh_unreachable`, `circuit_open`, ...). The run summary lists the backends and fallback reasons, and the UI
  badges show them.
- **Outbox (migration 0004).** Workspace deletion queues `POST /v1/cache/invalidate {namespace}` and
  `DELETE /v1/feedback/{mmr id}` for every ModelMesh request of the workspace, in the deleting transaction. A rating
  on a whole answer queues `user_thumbs` labels (`user_signal`, `attribution: workflow`, never training-eligible) for
  the run's requests. The worker drains the queue at most every 30 s and pauses while ModelMesh is down. A second,
  different rating of the same run is a `conflict` and is logged, not overwritten. Evaluation labels (M6) use the
  same poster, `ModelMeshOps.post_feedback`.
- **Combined benchmark**: `mezvuat-exp combined-benchmark` (`evaluation/combined_benchmark.py`). ModelMesh arms run
  without fallback and pin the routing policy for every task type. Not run yet; see
  `docs/eval/combined_benchmark/report.md`.

## Consequences

- Mezvuat runs without ModelMesh: public workspaces need a Groq key, private workspaces need Ollama. ModelMesh adds routing, caching and telemetry with no node changes.
- Start-up negotiation makes Mezvuat tolerant of older or newer 1.x ModelMesh servers, without silently dropping isolation or privacy fields.
- Under 429 Mezvuat degrades by waiting or by failing the step. It does not reroute.
- We maintain a small client generated from `openapi-v1.yaml`, with contract tests derived from it.

## Verification plan

- **Contract tests (respx)**, generated from `openapi-v1.yaml` schemas, covering:
  - every row of the fallback table, including 500, 503 `upstream_capacity`, and 429 never reaching the direct backend;
  - a stream with a pre-commit 503 (HTTP error, then fallback);
  - a stream with `stream_interrupted` and no `[DONE]` (discard, single retry);
  - a stream ending without `[DONE]` and without an error (treated as incomplete);
  - the `include_usage` final chunk with `choices: []`.
- **Negotiation tests:**
  - An extension missing from `GET /v1/contract` is not sent.
  - A missing `cache.namespace` or `routing.data_policy` causes `inference_contract_mismatch`.
  - An unknown task type falls back to `general`.
  - `min_quality` is sent only when calibration says `calibrated: true`.
- **Format tests:**
  - `correlation_id`, `namespace`, `context_fingerprint` and metadata keys/values match the contract patterns. This is a property-based test over random ULIDs and workspace ids.
  - The fingerprint changes when a chunk version or the workspace changes, and does not change when evidence is reordered.
- **Schema test:** every Pydantic response schema is in the portable subset (CI).
- **Judge test:** `evaluation_judge` requests always carry an explicit model and `fallback: "none"`.
- **Privacy test:** a workspace whose policy excludes the direct provider makes zero outbound requests to it, and ModelMesh requests carry `data_policy`.
- **M8 integration:** a full research run through a real ModelMesh. Then stop ModelMesh: the next calls fall back to direct (if policy allows), and the run completes with fallback badges.

## Sources (accessed 2026-11-08)

- Contract 1.0-rc2: §0, §1, §2 (extensions, metadata), §2.2–§2.5, Streaming, §3, §3.1, §4, §5, §7 changelog R1–R11 and C1–C8
- `openapi-v1.yaml`: `RequestMetadata`, `CacheOptions`, `ContractInfo`
- `CR_RESOLUTIONS.md`
- Groq: console.groq.com/docs/structured-outputs, console.groq.com/docs/your-data, console.groq.com/docs/rate-limits
- Gemini: aistudio.google.com/docs/terms, aistudio.google.com/docs/structured-output
