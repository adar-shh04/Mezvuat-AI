# 0008. LangGraph research workflow

Proposed · 2026-11-08 · Spec §1.3D, §3.3, §1.5 (LangGraph owns the workflow)

## Context

The workflow has eight stages and must be explicit and inspectable. Iteration is bounded, with no open-ended agent loops and no multi-agent design. Each LLM call carries a `task_type`, so ModelMesh can route it (contract §2.2).

## Options considered

1. **A LangGraph `StateGraph` with explicit nodes and conditional edges, no tool-calling agent.** Chosen.
2. A ReAct or tool-calling agent. It is opaque and its iteration count is unbounded, so the spec rules it out.
3. Plain Python functions without LangGraph. Inspectable, but the spec names LangGraph, and we would lose checkpoint and stream support.

## Decision

LangGraph **1.2.x** (MIT). LangChain is used only for `langchain-core` message types, where LangGraph needs them. No LangChain chains or agents.

### Graph

```
START → understand_query → plan_research → retrieve → assess_evidence
assess_evidence --(insufficient & iteration < MAX_ITER & budget ok)--> reformulate → retrieve
assess_evidence --(sufficient | limits hit)--> synthesize → verify
verify --(unsupported claims & resynth_count < 1)--> synthesize
verify --(ok | resynth limit)--> respond → END
any node --(fatal: retrieval_unavailable / inference_unavailable)--> fail → END
```

| # | Stage (§1.3D) | Node | LLM? | `task_type` | Output (structured, Pydantic) |
|---|---|---|---|---|---|
| 1 | Query understanding | `understand_query` | yes | `query_classification` | `QueryAnalysis{objective, subject, constraints, expected_answer_type, temporal_scope{as_of?, wants_latest?, compare_versions?}, needs_multi_doc}` |
| 2 | Research planning | `plan_research` | yes | `research_planning` | `ResearchPlan{subquestions[≤4], per-subq filters, success_criteria}` |
| 3 | Evidence retrieval | `retrieve` | **no** | — | hybrid + rerank per subquestion (ADR-0006); merges evidence into the registry (ADR-0009) |
| 4 | Evidence assessment | `assess_evidence` | yes | `evidence_assessment` (in contract 1.0-rc2. Sent as `general` only if `GET /v1/contract` does not list it; see ADR-0011) | `Sufficiency{per-subq covered: bool, missing: [..], rationale}` |
| 5 | Additional research | `reformulate` | yes | `query_rewriting` | `Reformulation{subq_id, new_queries[≤2]}`, only for uncovered subquestions |
| 6 | Synthesis | `synthesize` | yes | `evidence_synthesis` | `Draft{claims[{id, text, evidence_ids[], quotes[], kind: fact\|interpretation}], conflicts[], gaps[]}` |
| 7 | Verification | `verify` | partly | `citation_verification` (only for claims the deterministic + NLI checks cannot decide) | `Verification{per-claim status}` (ADR-0009) |
| 8 | Response generation | `respond` | yes, streamed | `final_response` | Markdown answer using **only** verified claims and their citation markers, plus a limitations section. Post-validated (ADR-0009). |

A cheap shortcut: if `understand_query` reports a single-hop factoid, `plan_research` may emit one subquestion equal to the query. This is still a node, so it still shows up in traces.

### State schema

`ResearchState`, a Pydantic model whose list fields use reducers:

- **Identity:** `run_id` (ULID, also the `correlation_id`), `workspace_id`, `user_id_hash`
- **Input:** `question`, `as_of: date | None`, `filters`, `retrieval_config_id`
- **Analysis:** `analysis: QueryAnalysis | None`, `plan: ResearchPlan | None`
- **Evidence:** `evidence: dict[EvidenceId, EvidenceItem]` (chunk_id, doc/version ids, page and section, text, scores, retrieval round), `retrieval_log: list[RetrievalCall]`
- **Loop control:** `iteration: int`, `resynth_count: int`, `sufficiency`
- **Output:** `draft`, `verification`, `answer`, `limitations: list[str]`
- **Budget:** `llm_calls: list[LLMCallRecord]` (task_type, backend, X-Request-ID, mmr id, tokens, cost, latency, cache status), `budget: Budget{llm_calls_left, tokens_left, deadline}`
- **Failure:** `status`, `error`

### Bounds

| Limit | Default | Where |
|---|---|---|
| Additional research iterations (`MAX_ITER`) | 2 | `assess_evidence` edge |
| Subquestions | 4 | `ResearchPlan` validator |
| New queries per reformulation | 2 | `Reformulation` validator |
| Re-synthesis after failed verification | 1 | `verify` edge |
| LLM calls per run | 12 | `Budget`, checked before every LLM node |
| Evidence items passed to synthesis | 20 (token-budgeted) | `synthesize` |
| Wall clock per run | 120 s | `Budget.deadline` |
| LangGraph `recursion_limit` | **set explicitly to 30** | `invoke` config. The LangGraph default is now 10,007, which is far too permissive to serve as a guard. |

When a limit is hit, the graph moves forward: insufficient evidence leads to synthesis, which states the gap explicitly. **It never loops further, and it never fabricates.**

### Runtime

- **Checkpointer:** `langgraph-checkpoint-postgres` 3.x (MIT, psycopg 3), so each run is inspectable step by step and can resume after a crash.
- **Retention:** checkpoints are pruned after N days (configurable).
- **Streaming to the client:**
  - `stream_mode=["updates", "custom"]` drives node progress events over SSE to the frontend (`/runs/{id}/events`).
  - `respond` streams answer tokens as `custom` events.
- **Structured nodes do not stream.** Groq does not support streaming together with structured output, and it is not needed for those nodes.
- **Prompt versioning:** every node prompt lives in `orchestration/prompts/*.md` with a version id. That id is recorded per LLM call and fed into `context_fingerprint` (ADR-0011).

## Consequences

- Fully inspectable runs, with predictable worst-case cost: 12 calls.
- Some quality lost on very hard multi-hop questions because of the bounds. That is intentional and measured in M6.
- Persisting checkpoints adds Postgres rows per run, so pruning is needed.

## Verification plan

- **Graph tests with a fake `LLMClient`** that returns scripted structured outputs:
  - Every edge condition is exercised.
  - `MAX_ITER` stops the loop.
  - A budget exhaustion path reaches `respond` with limitations.
  - Retrieval failure reaches `fail`.
- **Static test:** the graph contains no cycles that bypass the iteration counters (checked by walking the compiled graph).
- **Trace test:** every LLM call record has a `task_type` from the contract vocabulary and a unique `X-Request-ID`.

## Sources (accessed 2026-11-08)

- pypi.org/pypi/langgraph (1.2.20)
- docs.langchain.com/oss/python/langgraph/graph-api (recursion_limit default 10,007 since 1.1.4)
- docs.langchain.com/oss/python/langgraph/streaming
- pypi.org/pypi/langgraph-checkpoint-postgres (3.2.0)
- console.groq.com/docs/structured-outputs (no streaming with structured outputs)
