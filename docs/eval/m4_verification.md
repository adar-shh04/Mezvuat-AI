# M4 verification report: LangGraph workflow and citations (direct provider)

Executed 2026-11-14 on the dev machine (Windows 11, Python 3.13), branch `phase-2`. The commands run from `backend/` after `. ../scripts/dev-env.sh`.

## Quality gates

| Command | Result |
|---|---|
| `uv run --no-sync ruff check .` | All checks passed |
| `uv run --no-sync ruff format --check .` | all files formatted |
| `uv run --no-sync mypy` | no issues in 82 source files (`inference`, `citations` and `orchestration` are strict) |
| `uv run --no-sync lint-imports` | 6 contracts kept, 0 broken |
| `uv run --no-sync pytest -q` | 201 passed, 15 skipped (the 15 are integration tests; the compose Postgres and Qdrant were not running) |

## Acceptance criteria

**D7: bounded LangGraph.** `tests/unit/test_research_graph.py` runs the real compiled graph with a scripted `LLMClient`:

| Criterion | Test |
|---|---|
| Every edge | `test_single_hop_happy_path_visits_every_stage_once`, `test_multi_hop_plan_reformulates_once_then_answers`, `test_reformulation_without_new_queries_goes_to_synthesis`, `test_failed_verification_triggers_one_resynthesis`, `test_retrieval_failure_goes_to_fail`, `test_inference_unavailable_goes_to_fail_and_still_records_the_call` |
| `MAX_ITER` stops the loop | `test_max_iter_stops_the_research_loop` (exactly 2 reformulations, 3 assessments) |
| Budget exhaustion reaches `respond` with limitations | `test_budget_exhaustion_reaches_respond_with_limitations` (call budget), `test_deadline_exhaustion_skips_llm_steps_and_abstains` (wall clock) |
| Retrieval failure reaches `fail` | `test_retrieval_failure_goes_to_fail` |
| `recursion_limit` set explicitly | `test_recursion_limit_is_set_explicitly` (30) |
| No cycle bypasses the counters | `test_graph_cycles_all_pass_through_a_bounded_counter` walks the compiled graph: the only cycles are `retrieve → assess_evidence → reformulate` and `synthesize ⇄ verify` |
| Trace: task types from the contract vocabulary, unique `X-Request-ID` | asserted in the happy-path test |

**D8 (partial): verifiable citations.**

- `test_every_citation_marker_resolves_to_a_verified_span_on_the_right_page`: every `[E#]` in the final answer is a registry id. Each of its citations is a verbatim substring of the chunk at the stored absolute offsets, and the page comes from the version's page map (a quote past the page break resolves to page 2).
- Fabricated id rejected: `test_fabricated_evidence_id_is_rejected` (unit) and `test_failed_verification_triggers_one_resynthesis` (graph, `E99`). Ids the synthesis prompt did not show are also rejected: `test_claims_citing_ids_the_synthesis_prompt_did_not_show_are_rejected`.
- Paraphrase rejected, hyphenation accepted, unknown or pseudo markers replaced: `tests/unit/test_citations.py`, `test_unknown_marker_in_the_answer_is_replaced_by_the_verified_claims`.

**D11 (direct half): a full run with only `GROQ_API_KEY`.** `tests/unit/test_direct_groq_run.py` builds the client from settings that contain only a Groq key. It runs the full graph against a mocked Groq endpoint that uses Groq's wire format (respx blocks every other request) and gets a cited answer after 4 Groq requests. **This is not a live Groq call.** The Groq free quota was reserved for the M3 pre-labelling job, so live checks used Ollama (below).

**API.** `tests/unit/test_runs_api.py` covers `POST /runs` (202), the SSE progress stream (node events, streamed answer deltas, terminal event, `Last-Event-ID` resume), `GET /runs/{id}` with persisted `llm_calls`, failure paths, workspace scoping and the after-restart path. It uses an in-memory run store.

## Live check with a local model

`scripts/smoke_run.py` parses a real demo-corpus PDF with the production parser and chunker, searches it with in-memory BM25, and runs the real graph through Ollama 0.40.16 (`llama3.2:3b`) under `local_only`:

```
uv run --no-sync python ../scripts/smoke_run.py --backend ollama --model llama3.2:3b \
  --pdf X:/mezvuat-cache/data/corpus/arxiv/2606.06474v2.pdf \
  --question "How does self-augmenting retrieval work for diffusion language models?"
```

Observed in two runs (about 60 s each):
- All 8 stages ran, including 2 reformulation rounds (stopped by `MAX_ITER`) and 1 re-synthesis.
- 9 LLM calls, all `ok`, and structured output was valid on every call.
- With this 3B model, most claims failed the quote check and were dropped and listed under limitations. In the second run the model's answer had no citation markers, so it was replaced by the deterministic rendering of the one verified claim.
- The single remaining citation re-matched the PDF text exactly (`[E7] p.4 match=True`).

This shows that the safeguards work on a weak model. It says nothing about answer quality, which M6 measures.

**One live Groq run.** It was run once, at the end of the session, after the pre-labelling job had finished. Same PDF and question, `--backend groq`, policy `any` (public corpus, decision D1):

- The run completed in 7.4 s with 5 calls: `gpt-oss-20b` for analysis and assessment, `gpt-oss-120b` for synthesis (twice, because the first draft had a claim that failed verification) and for the response.
- About 12,700 tokens in total. Every call returned `ok`, with no 429.
- The final draft had 3 supported claims, and the model's answer kept only valid markers.
- All 4 citation spans re-matched the PDF text exactly.

The Sources section said "dates unknown", because the smoke script has no temporal source.

## Verified afterwards (2026-11-14, coordinator)

With the compose Postgres and Qdrant started (`docker compose -f infra/compose.yaml up -d postgres qdrant`), `uv run --no-sync pytest -q -m integration` at `c7827e8` gave **16 passed, 0 skipped**. This ran migration `0002` against a live database and exercised `SqlRunStore`, `SqlCallRecorder`, `PostgresPageMap` and the `PostgresSaver` checkpointer (`tests/integration/test_runs_persistence.py`, 2 tests), plus the 14 ingestion end-to-end tests. The tests use the `mezvuat_test` database and a per-process Qdrant collection, so the dev data was not touched.

## Not verified

- **End-to-end `/runs`** against real Postgres, Qdrant and the embedding model.
- **Checkpoint retention pruning** is not implemented.

# M5 progress (parts 1 and 2), same session

Gates after part 2: ruff clean, mypy clean on 84 files, 7 import contracts kept, **241 passed, 16 skipped**. The skipped tests are the integration tests, because no Postgres or Qdrant was running.

**D9: temporal (`tests/unit/test_temporal.py`).**
- `ingested_at` is not a date kind, so it cannot decide "latest".
- Same-year year-precision and same-month month-precision dates do not resolve an as-of query.
- Unknown dates and same-day ties stay unresolved.
- `effective_from` and `effective_to` take precedence.
- In the graph, an as-of question uses only the version that applied on that date. Unresolved versions and undated documents are named under limitations.
- Latest-version questions use the latest flag.
- Answers show each cited version's dates by kind in a Sources section.

**D8: adversarial (`tests/unit/test_adversarial.py`).** The suite has 22 hostile documents, run against a scripted model that obeys them. In all 22 cases, 0 unknown citations and 0 prompt shingles reach the answer. Flag rate:
- 20/22 hostile documents (the 2 misses are deliberately subtle);
- 0/8 benign controls;
- 16/2,261 chunks (0.71%) of 60 real demo-corpus papers. This was measured with the production parser and chunker and is not hand-checked.

**Not done in M5:**
- the NLI verifier and its threshold calibration, which needs a small hand-labelled claim/evidence set from the product owner;
- the LLM-judge fallback;
- the conflicting-evidence section, which needs NLI contradiction scores;
- the temporal question set of 30 or more questions with gold answers.

Until NLI exists, claims are verified by quote match only, and every answer says so.
