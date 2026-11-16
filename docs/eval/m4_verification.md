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

# M5 part 3: NLI verifier, LLM-judge fallback and calibration tooling (2026-11-16)

Executed on the dev machine, branch `phase-2`, while the ModelMesh benchmark was running. No Groq call was made, and no Ollama call either: the 4 GB GPU held the benchmark's `qwen3:4b`. All NLI work ran on CPU with `CUDA_VISIBLE_DEVICES=-1`. Details are in the ADR-0009 amendment of 2026-11-16.

## Quality gates

| Command | Result |
|---|---|
| `ruff check .` / `ruff format --check .` | All checks passed / 134 files already formatted |
| `mypy` | no issues in 92 source files |
| `lint-imports` | 7 contracts kept, 0 broken |
| `python ../scripts/export_openapi.py --check` | `frontend/openapi.json` is up to date |
| `pytest -q` with the compose Postgres and Qdrant up | **328 passed, 0 skipped** (281 before); **339 passed** after the v2 pair rules (11 more skip-rule tests). The live-model test ran because the model is in the local cache. |
| `npm run typecheck`, `npm run lint`, `npm run build` | no errors, no problems, compiled |
| `npx playwright test` (dev server, mocked backend) | **3 passed** |

The new tests are in `tests/unit/test_nli_verifier.py` (20) and `tests/unit/test_nli_calibration_tools.py` (27).

## What is verified

- **NLI in the graph,** using a rule-based NLI fake:
  - an entailed claim is `supported` with method `nli`;
  - a contradicted claim gets the one re-synthesis, is dropped, and is listed under Conflicting evidence with its span, page and dates;
  - a claim whose sources disagree has `between_documents` set;
  - an undecided claim is kept as `partially_supported`;
  - a model failure, or too little run time left, falls back to the quote check and the answer says so.
- **The real model:** the pinned model loads on CPU with the registered label order. On three hand-written pairs it labels the entailment and the contradiction correctly. This is a smoke test, not an accuracy figure.
- **LLM judge,** with scripted fakes only:
  - it is off by default;
  - a verdict counts only with a verbatim quote;
  - a contradiction it finds is listed as a conflict;
  - it never spends the response's call;
  - its calls are capped;
  - the request carries `citation_verification`, temperature 0 and the run's data policy.
- **Frontend:** the conflicts panel shows the contradicted claim and its quote, and opens the span on its page in the evidence viewer. The claims panel says which checks ran and that the thresholds are provisional.
- **Review modes:**
  - `review nli` saves, resumes and undoes labels;
  - no suggestion appears on the review screen;
  - `val`, `test`, `temporal` and `nli` each reach their own tool and file.

## Calibration set

- **Location:** `data/domain/nli/v1/pairs.jsonl`, 150 pairs from 29 val-family papers.
- **By expected label (construction prior, not gold):** 50 supported, 50 contradicted, 50 neutral.
- **By method:**

| Method | Pairs |
|---|---|
| `sentence_verbatim` | 12 |
| `sentence_rewrite` | 38 |
| `perturb_number` | 15 |
| `perturb_negation` | 13 |
| `perturb_antonym` | 16 |
| `perturb_swap` | 6 |
| `same_passage_far` | 20 |
| `same_paper_other_passage` | 18 |
| `other_paper` | 12 |

- **Review command:** `.\scripts\review.ps1 nli`. Nothing is labelled yet.

### Version 2 of the set, rebuilt before any labelling

The coordinator reviewed v1 and found about 10 of the 50 perturbations that changed no meaning or produced broken text:
- numbers inside citation brackets;
- version, licence, dataset or venue numbers;
- a digit taken out of a grouped number;
- "above" swapped for "below";
- name swaps that were not comparisons;
- double negatives.

v2 (`nli-pairs-v2`) adds a skip rule for each, plus a few more I found while reading every v2 perturbation: dash ranges, ratios, metric names in swaps, idioms such as "no longer" and "how many", and "have" or "do" as the main verb. Each rule has a unit test that uses the cases from the review.

**Result:** still 150 pairs from 29 papers, with 50/50/50 by expected label and the same per-method counts as above. 147 of the 150 pairs differ from v1, because the filters changed the candidate pool. `review.human.jsonl` did not exist when the file was overwritten.

**Rule counts.** Each rule is counted over the whole pool of 5,196 candidate sentences: per number occurrence for the number rules, per sentence for the others. The numbers are in `manifest.json` under `rejection_audit`.

| Rule | Rejected |
|---|---|
| number in a citation bracket | 602 |
| unmeasured number in a sentence with no comparison | 607 |
| section, table or figure number | 284 |
| year | 147 |
| end of a dash range | 110 |
| number right after a name | 53 |
| no valid replacement | 44 |
| version, licence, dataset or venue | 17 |
| antonym: deictic or positional word | 104 |
| antonym: fixed term, idiom or discourse marker | 66 |
| negation: sentence already negative | 467 |
| negation: a negating word at the insertion point | 187 |
| negation: "have" or "do" as the main verb | 60 |
| swap: no comparison in the sentence | 5,032 |
| swap: no name on both sides of the comparison in one clause | 142 |
| swap: names contain each other | 5 |
| swap: first name is not the subject | 2 |
| swap: a name is repeated | 2 |

**Perturbations possible after these rules:** number 476, negation 1,721, antonym 1,012, swap 13.

**Known extraction issue: glued text.** PyMuPDF drops the spaces in some text of the RAG-Stack paper, for example "retrievalandgeneration:theretrieverruns". In the whole corpus, 53 of 10,125 chunks contain a run of 28 or more letters:
- 37 of RAG-Stack's 94 chunks;
- most of the others contain only long CamelCase identifiers such as `MultipleNegativesRankingLoss`;
- one train-split paper has a short glued run.

Among the val chunks the builder can use, 34 are excluded for this reason, from 3 papers. The ingestion pipeline does not fix the glued text, so retrieval and quote matching on those chunks see it as it is.

## CPU latency

Measured on the 150 v1 pairs, with 6 torch threads, while the benchmark shared the machine:

| Measure | Result |
|---|---|
| One pair per call | median 231 ms, p95 586 ms |
| Batched, 16 pairs | 155 ms per pair |
| Model load | 33 s |

## Not verified

- **Calibration.** No thresholds have been calibrated, because no owner labels exist. Answers use the provisional rule (τ_e = τ_c = 0.5) and say so. Calibration is tested on synthetic labels only.
- **NLI accuracy** on this corpus is unknown until the owner's labels exist.
- **A live judge call.** The judge has never been run against a live model.
- **An end-to-end `/runs` call** with the real NLI model behind the API.
- **The compose stack.** NLI is off there by default: the model adds about 0.7 to 0.85 GB to the process (measured natively), and the API's 1200m limit was not resized or tried.
