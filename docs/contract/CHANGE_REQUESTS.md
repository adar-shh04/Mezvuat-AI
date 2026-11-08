# Change requests from Mezvuat AI against the ModelMesh Inference API, Contract v1 DRAFT 0.1

- **Raised by:** the Mezvuat agent, 2026-11-08.
- **Reviewed against:** `docs/contract/inference-api-v1.md`, a read-only mirror. I have not edited it.
- **Process:** the coordinator reconciles these requests with the ModelMesh agent, and ModelMesh owns the contract. Every request below is an additive, optional change, so v1 compatibility is preserved, unless it says otherwise.
- **Priority:**
  - **P1:** Mezvuat needs it before M8 integration.
  - **P2:** wanted, with a workaround in place.
  - **P3:** nice to have.

## Per-section verdict

| Section | Verdict |
|---|---|
| §0 Principles | **CR-12** (P2) |
| §1 Auth and correlation | **CR-08** (P2). Otherwise no changes are needed: Mezvuat will send `X-Request-ID`, `traceparent`, and store `X-ModelMesh-Request-ID`. |
| §2 Request extensions | **CR-07** (P2), **CR-10** (P2) |
| §2.1 Routing policies | No changes needed. Mezvuat uses `cost_optimized` and `quality_balanced` by default and never sends `min_quality` until CR-11 is resolved. |
| §2.2 `task_type` | **CR-01** (P1) |
| §2.3 Cache correctness | **CR-04** (P1), **CR-05** (P2) |
| Response (non-streaming) | **CR-02** (P1) |
| Streaming | **CR-03** (P1) |
| §3 Errors | **CR-06** (P1) |
| §4 Other endpoints | **CR-09** (P2). `/v1/embeddings`: no changes needed, Mezvuat will not use it in v1 (ADR-0006). |
| §5 Feedback | **CR-13** (P1) |
| §6 Open questions | Mezvuat's positions are in **CR-14**. |

---

## CR-01 (P1). `task_type`: add `evidence_assessment` and `evaluation_judge`

- **Why:**
  - Mezvuat's workflow (ADR-0008) has an evidence-sufficiency node, a short structured judgment with a different profile from classification or synthesis.
  - Offline RAGAS and judge calls (ADR-0012) should not pollute router training data for `citation_verification` or `general`.
- **Proposal:** add these two values to the v1 enum:

  | Value | Use |
  |---|---|
  | `evidence_assessment` | Structured sufficiency judgment over evidence summaries |
  | `evaluation_judge` | LLM-as-judge for offline evaluation. ModelMesh may want to exclude it from routing-quality labels. |

- **Mapping Mezvuat will use:**

  | Workflow node | `task_type` |
  |---|---|
  | `understand_query` | `query_classification` |
  | `plan_research` | `research_planning` |
  | `assess_evidence` | `evidence_assessment` |
  | `reformulate` | `query_rewriting` |
  | `synthesize` | `evidence_synthesis` |
  | `verify` | `citation_verification` |
  | `respond` | `final_response` |
  | eval | `evaluation_judge` |
  | domain query generation | `general` |

- **Until it is accepted:** Mezvuat sends `general` for these two and puts the intended value in `metadata.tags.task_type_intended`.

## CR-02 (P1). Structured output guarantees and reporting

- **Why:** citation verification and every structured node depend on schema-valid JSON. Providers differ:
  - Groq `strict` is constrained decoding.
  - Others may need emulation.

  Mezvuat needs to know which one happened.
- **Proposal:**
  1. When `response_format.type = "json_schema"` and `required_capabilities` contains `json_schema`, ModelMesh **validates** the output against the schema before returning 200.
  2. Add a new capability token `json_schema_strict`, meaning native constrained decoding, so callers can require it. Plain `json_schema` allows "emulated + validated".
  3. Add a response field: `modelmesh.structured_output: {"mode": "native_strict" | "native" | "emulated", "validated": true, "repair_attempts": 0}`.
  4. Add a new error, `502 output_validation_failed`, `retryable: true`. It is returned when validation fails after ModelMesh's repair budget, with `details.validation_errors` truncated and containing no content.
  5. Document the **portable JSON Schema subset**, for example: all properties required, `additionalProperties: false`, no `minLength`/`uniqueItems`/`pattern`. Mezvuat's schemas will stay inside it.
- This also answers Open Question 3 (see CR-14).

## CR-03 (P1). Streaming the final response

1. **Headers at stream start.** `X-ModelMesh-Request-ID`, `X-Request-ID` and `X-ModelMesh-Contract` must be sent as response **headers** before the first event, so Mezvuat can correlate a stream that dies mid-way.
2. **In-stream error event shape.** Same envelope as §3:

   ```
   data: {"error": {"type", "code", "message", "request_id", "retryable", "details"}}
   ```

   It is followed by `data: [DONE]`. A final `modelmesh` object (attempts, usage so far, `usage.source`) should also be emitted when possible.
3. **Client disconnect.** State that a client disconnect cancels the upstream request, and that usage is still accounted.
4. **`finish_reason`.** Normalize it to `stop | length | content_filter | error` in the final chunk.
5. **Structured plus streaming.** State whether `stream: true` with `json_schema` is allowed. Mezvuat proposes it is allowed only if the candidate supports it, otherwise `422 no_eligible_model`. Mezvuat itself will not stream structured calls.

## CR-04 (P1). `context_fingerprint`: opaque, validated, plus a caller cache namespace

- **Why:** §2.3 currently prescribes Mezvuat's internal formula ("hash of retrieved chunk ids + document version hashes"). That is the consumer's business; the contract should only define the field.
  - Also, the ModelMesh tenant is the API key, but Mezvuat has many **workspaces** under one key. Cache reuse across workspaces would leak content across them (spec §5.3).
- **Proposal:**
  1. Make the definition normative and opaque: `context_fingerprint` is a caller-computed string matching `^sha256:[0-9a-f]{64}$`.
     - ModelMesh never interprets it.
     - It is part of the exact key (already the case) and must be identical for semantic hits (already the case).
  2. Move the Mezvuat formula to a non-normative example. Mezvuat's actual formula is in ADR-0011: `workspace_id`, `prompt_version`, and sorted `(chunk_id, version text hash, chunker_version)`.
  3. Add an optional `cache.namespace` (string, ≤ 128 chars). It is included in **both** exact and semantic keys, and hits never cross namespaces within a tenant. Mezvuat sets it to the workspace id, hashed.
  4. State whether a semantic lookup without `context_fingerprint` is a 400 (proposal: yes, `invalid_request`).

## CR-05 (P2). Clarify the cache default for non-deterministic requests

- **Problem:** §2.3 says that with `temperature > 0` and no `seed`, a cache read is "allowed only if the caller opted in (cache.mode set explicitly)". But the default mode is `exact`, so it is ambiguous what happens when `cache` is omitted.
- **Proposal:** when `cache` is omitted and the request is non-deterministic, the effective mode is `off` for reads, and the response reports `cache.status: "bypass"`.
- **What Mezvuat will do regardless:** send `cache.mode: "off"` for `final_response`, and `"exact"` for temperature-0 structured calls.

## CR-06 (P1). Error and fallback semantics

1. **429 and consumer fallback.** §3 says direct-provider fallback triggers "not on 4xx".
   - Mezvuat proposes that **429 is an explicit exception**: honour `Retry-After` within the caller's deadline, then the caller may fall back.
   - `Retry-After` must be present on every 429, in seconds.
2. **Machine-readable `details` per type:**

   | Type | Proposed `details` |
   |---|---|
   | `no_eligible_model` | `{"violated_constraints": [{"constraint": "max_cost_usd", "best_available": 0.031}]}` |
   | `upstream_error` / `upstream_timeout` | `{"attempts": [...]}`, same shape as `modelmesh.attempts` |
   | `budget_exceeded` | `{"scope": "key" \| "tenant", "resets_at": "..."}` |

3. **Echo `client_request_id`** in the error body as well as `request_id`.
4. **Error for pre-routing failure.** If ModelMesh fails before assigning `mmr_…`, say what `request_id` contains (proposal: still an `mmr_` id, always assigned on receipt).
5. **`X-Request-ID` is not an idempotency key.** Mezvuat sends a new one per attempt. State this, or add an explicit optional `Idempotency-Key` header if ModelMesh wants dedupe.
6. Mezvuat confirms it will **not** fall back on 400, 401, 402, 403 or 422. That would silently violate policy.

## CR-07 (P2). Hard timeout separate from latency budget

- **Proposal:** add `routing.timeout_ms`, an enforced end-to-end deadline that returns `504 upstream_timeout`. `max_latency_ms` stays a routing budget on *predicted* latency only.
- **Default:** if `timeout_ms` is absent, 2 × `max_latency_ms` (this resolves Open Question 2).
- **Why:** Mezvuat's workflow has per-node deadlines (ADR-0008) and must not wait for an upstream request it has already given up on.

## CR-08 (P2). Correlation

1. Document limits:
   - `correlation_id`: ≤ 128 chars, `[A-Za-z0-9_\-:.]`.
   - `tags`: string → string, ≤ 16 keys, values ≤ 256 chars.
   - `end_user_id`: ≤ 128 chars.
2. State that `metadata` is stored with execution metadata and is returned by `GET /v1/requests/{id}`, but is never used for routing except `task_type`.
3. Confirm that `traceparent` continuation means ModelMesh spans become children of the caller's span, so Langfuse and ModelMesh OpenTelemetry traces share a trace id.

## CR-09 (P2). List executions by correlation id

- **Proposal:** `GET /v1/requests?correlation_id=<id>&limit=` is tenant-scoped and returns the execution metadata objects, with no content.
- **Why:** the combined benchmark (spec §6) joins a Mezvuat research run to all its ModelMesh executions. Mezvuat stores the mmr ids anyway, so this is P2.

## CR-10 (P2). `data_policy` semantics per provider account tier

- **Why:** whether a provider trains on data depends on the **account tier**, not just the provider. For example, Gemini free-tier prompts are used to improve Google products, while paid-tier prompts are not (aistudio.google.com/docs/terms, accessed 2026-11-08).
- **Proposal:**
  - Define `no_training_providers` at the configured provider-account level.
  - Expose a per-model `data_training: true | false | unknown` in `GET /v1/models`.
  - `unknown` is treated as not eligible under `no_training_providers`.
- **What Mezvuat will send:** `data_policy: "no_training_providers"` by default for any call that contains document content.

## CR-11 (P3). `min_quality` calibration

- Mezvuat will not send `min_quality` until its scale is documented per `task_type`, including what 0.7 means and on which benchmark.
- **Proposal:** publish calibration metadata via `GET /v1/models` or a docs page, with `router_version`. This addresses Open Question 1.

## CR-12 (P2). Version negotiation vs "unknown extension fields rejected (422)"

- **Problem:** §0 rejects unknown extension fields with 422. That makes a newer client sending a 1.3 field to a 1.1 server fail hard.
- **Proposal:** `GET /v1/health` (or a new `GET /v1/contract`) returns `{"contract_version": "1.x", "supported_extensions": [...]}`. Mezvuat reads it at start-up and only sends fields the server supports.
- Keeping the 422 is fine once negotiation exists.

## CR-13 (P1). Feedback payload

The current shape is good. Requested additions are all optional, apart from defining responses:

1. **Fields:**

   | Field | Purpose |
   |---|---|
   | `metric_version` | Which version of the metric produced the score |
   | `eval_run_id` | Mezvuat MLflow run id, for provenance |
   | `dataset_version` | |
   | `labelled_at` | RFC 3339 |
   | `higher_is_better` | Default true |
   | `attribution` | `direct` (this request's output was evaluated) or `workflow` (an end-to-end score shared by several requests) |

   Router training should probably use only `direct`.
2. **Batch.** `POST /v1/feedback` should accept `{"items": [...]}`, up to 500, with per-item results, because eval runs produce hundreds of labels.
3. **Responses:**
   - 201 created, or 200 on an idempotent replay.
   - 404 for an unknown or foreign `request_id`, without revealing which.
   - 422 for an invalid score range.
4. **Metric vocabulary.** Documented in the contract with initial entries:

   | Metric | Source |
   |---|---|
   | `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` | RAGAS |
   | `citation_validity`, `citation_support_rate`, `temporal_accuracy`, `schema_valid` | Mezvuat custom |
   | `user_thumbs` | User signal |

5. **Deletion.** `DELETE /v1/feedback/{request_id}?metric=&evaluator=` supports data-deletion requests (spec §5.3).
6. **Content.** Confirm that `notes` must not contain prompt or document content. Mezvuat will never send content there.

## CR-14. Mezvuat positions on the §6 open questions

| Open question | Mezvuat position |
|---|---|
| 1. `min_quality` calibration | CR-11. Mezvuat does not use it until it is documented. |
| 2. Timeout semantics | CR-07. Separate `timeout_ms`, defaulting to 2 × `max_latency_ms`. |
| 3. Structured output on providers without native `json_schema` | CR-02. Allow emulation with validation and repair, but report `mode`, and let callers require `json_schema_strict`. Never return unvalidated JSON with a 200 when `json_schema` is required. |
| 4. Content retention | Agree with **off** by default. Mezvuat will not enable it for workspaces with user documents. |

---

## Resolution status (Mezvuat, 2026-11-08, against 1.0-rc2)

I reviewed `CR_RESOLUTIONS.md`, the rc2 changelog (R1–R11), and `openapi-v1.yaml`. **Mezvuat accepts every resolution, including the four rejected sub-items. No CR is still disputed, and no CR-15 is raised.** ADR-0011 has been amended to match rc2.

| CR | ModelMesh disposition | Mezvuat status | Notes |
|---|---|---|---|
| CR-01 | accepted | **Accept** | Uses `evidence_assessment` and `evaluation_judge`. The interim `task_type_intended` key is kept only if `GET /v1/contract` does not list a value. |
| CR-02 | accepted with modification | **Accept** | Default is `json_schema` (native only). `json_schema_strict` for `citation_verification` and `evidence_synthesis`, with one retry on `json_schema` after a 422. Schemas stay inside the portable subset; length limits are enforced client-side. `native_best_effort` naming is fine. |
| CR-03 | accepted with modification | **Accept, including the rejection of 03.2 and 03.4** | A missing `[DONE]` means incomplete, which is safer than `[DONE]` after an error. Failures are signalled by the error event, not `finish_reason`. Pre-commit failures are handled as HTTP errors. `include_usage: true` is set. |
| CR-04 | already in rc1 | **Accept** | Fingerprint is `sha256:<64 lowercase hex>`. `cache.namespace` = HMAC of the workspace id, sent on every call. The workspace is also hashed into the fingerprint as defence in depth. |
| CR-05 | already in rc1 | **Accept** | Mezvuat sets `cache.mode` explicitly anyway. |
| CR-06 | accepted with modification | **Accept, including the rejection of 06.1** | Mezvuat adopts §3.1. Fallback happens only on network errors and 500/502/503/504. On 429, back off for `Retry-After` within the node deadline, otherwise fail the step; never fall back. Rationale: spec §4.3 asks for fallback when ModelMesh is *unavailable*, and a key-level 429 is not unavailability. Upstream exhaustion is 503 `upstream_capacity`. The list form of `violated_constraints` will be parsed. |
| CR-07 | already in rc1 | **Accept** | `timeout_ms` = the node's remaining deadline. |
| CR-08 | accepted | **Accept** | Flat metadata. `correlation_id` = `run_<ULID>`, validated against `[A-Za-z0-9_:.-]{1,128}`. |
| CR-09 | already in rc1 | **Accept** | Used by the M8 combined benchmark. |
| CR-10 | accepted with modification | **Accept** | Mezvuat uses the `data_use` field name. The product-owner decision on which providers are eligible is open (ADR-0011, options A–E). |
| CR-11 | accepted with modification | **Accept** | `GET /v1/routing/calibration` is read at start-up. `min_quality` is sent only where `calibrated: true`. |
| CR-12 | accepted with modification | **Accept** | `GET /v1/contract` is read at start-up, and only the listed extensions are sent. A missing `cache.namespace` or `routing.data_policy` is treated as a hard incompatibility; the call is not sent without them. |
| CR-13 | accepted with modification | **Accept, including the rejection of 13.3** | Error code 400 is fine. Uses `/v1/feedback/batch` with `metric_version`, `attribution`, `eval_run_id`, `dataset_version` and `accepted`. A 409 conflict is logged and never overwritten. |
| CR-14 | already in rc1 | **Accept** | |

### Clarification request (not a dispute)

- **CQ-1: `cost_optimized` default `min_quality` while uncalibrated.**
  - §2.1 says `cost_optimized` applies a policy default `min_quality` of 0.6 when it is unset.
  - §2 says that when `min_quality` is set, models without calibrated evidence are ineligible.
  - §2.4 says every task type reports `calibrated: false` until M6.
  - **Question:** does the policy default act as a hard constraint before calibration exists? If so, every `cost_optimized` request would return 422.
  - **Mezvuat's assumption:** it does not; it is only a warning while uncalibrated.
  - Until this is confirmed, Mezvuat uses `quality_balanced` for every task type (ADR-0011).
