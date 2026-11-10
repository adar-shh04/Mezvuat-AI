# 0007. Retrieval evaluation dataset and reranker fine-tuning protocol

Proposed · 2026-11-08 · Spec §1.3C, §3.4 Loop A, §6 ("labelled retrieval evaluation dataset", "experiments reproducible")

## Context

The spec says not to assume fine-tuning helps. Before any claim, three things must exist:
- a credible baseline,
- a labelled dataset,
- an improvement on held-out data.

Evaluation data must be separate from training data. This ADR fixes the datasets, splits, metrics, tracking and the decision rule **before** any experiment runs, so the analysis is effectively pre-registered.

## Options considered

- **Public benchmark only.** Credible and comparable, but it says nothing about our domain or about our chunking and temporal features.
- **Domain synthetic set only.** Relevant to us, but biased toward the generator and with no external anchor.
- **Both (chosen):** a public benchmark to sanity-check the implementation and measure generalization, plus a domain set with human-verified test labels.
- **Labelling method for the domain set:**
  - Fully human: too costly.
  - Fully LLM-judged: not trustworthy as a test set.
  - **Chosen:** LLM-silver labels for train, human labels for val and test.

## Decision

### A. Public benchmarks (BEIR format; licenses checked 2026-11-08)

| Dataset | Corpus / test queries | Train split | License | Use |
|---|---|---|---|---|
| **SciFact** | 5,183 / 300 | yes: 809 queries, 919 qrels | claims CC BY 4.0; abstracts ODC-By 1.0 (allenai/scifact LICENSE.md) | **Primary public benchmark**: baseline for all 4 configs, plus optional extra fine-tuning data |
| **SCIDOCS** | 25,657 / 1,000 | none | repo LICENSE CC BY 4.0; whether it covers the data is UNVERIFIED | Zero-shot generalization and forgetting guardrail |
| NFCorpus | 3,633 / 323 | yes | "free to use for academic purposes" | Optional, local academic evaluation only, never redistributed |

- The BeIR/* Hugging Face tag (cc-by-sa-4.0) is **not** treated as authoritative. The upstream terms are.
- Datasets are downloaded by a script into git-ignored `data/beir/`, with checksums recorded. Nothing is committed.
- **Leakage between SciFact train and test.** The two splits share a corpus, which is standard BEIR. Our fine-tuning on SciFact train uses a **doc-disjoint variant**: any train query whose positive documents are also positives for a test query is dropped. We report both the doc-disjoint and the standard variants.

### B. Domain dataset (from the demo corpus, ADR-0013)

1. **Freeze the corpus.**
   - Commit a manifest of arXiv ids with all versions, licenses and SHA-256 of the files.
   - `dataset_version` = hash of the manifest plus the label files.
2. **Split before generating anything.** Splits are by **document family**, never by chunk:
   - A family is every version of an arXiv id.
   - Families are also merged across ids when MinHash over chunk text (Jaccard ≥ 0.8, threshold to be set at M3) shows substantial shared text.
   - Ratio is **60/20/20** train/val/test, seeded and stratified by arXiv primary category and year.
   - The split file is committed.
3. **Query generation** (offline, through the inference layer with `task_type=general`, recorded with model id).
   - For sampled non-reference chunks of each split's own families, generate 1–2 queries of a requested type: factoid, conceptual, terminology, cross-document comparison, or temporal (for example "as of v1…").
   - **Lexical bias control.** The prompt forbids copying more than 5 consecutive words. We compute query–chunk longest-common-substring and token overlap, and report the distributions. Queries above a threshold are rewritten or dropped, so BM25 does not get an artificial advantage.
4. **Dedupe** with embedding cosine similarity ≥ 0.95 across **all** splits. A near-duplicate that crosses splits is removed from train (test and val win).
5. **Relevance labels by pooling.** The pool is the union of the top-10 from every system under study: bm25, dense with two embedders, hybrid_rrf, hybrid_rrf_rerank. The source chunk is always added. Grades are 0 (not relevant), 1 (partially relevant), 2 (answers the query).
   - **Test and val: human judges** follow a written guideline (`docs/eval/labeling_guide.md`, M3 deliverable).
     - A second annotator labels a 20% overlap, and we report Cohen's κ.
     - Disagreements are adjudicated.
     - The source chunk is **not** assumed relevant.
   - **Train: LLM-judge silver labels.** The judge model is different from the generator where possible. A human audit of 100 random pairs estimates the silver label precision, which we report.
   - Unjudged documents count as non-relevant. We report **judged@10** coverage for every system, so pool bias is visible.
6. **Target sizes:**
   - test ≥ 150 queries;
   - val ≥ 100;
   - train 1–3k silver.

   About 12 pooled passages per query (EST) means about 3,000 human judgments for val and test. **Who labels is an open question for the product owner.**
7. **Freezing.** Test labels are frozen: their content hash is logged on every evaluation. Any edit makes a new `dataset_version`. All tuning (k, depths, chunk size, learning rate, epochs) uses **val only**. Test is evaluated once per pre-registered comparison.

### C. Metrics

- **Relevance metrics:**
  - Recall@{10, 50, 100}
  - Precision@{5, 10}
  - MRR@10
  - **nDCG@10 (primary**, graded gains 0/1/2)
- **Efficiency metrics:**
  - Latency p50/p95 per stage (lexical, dense, fusion, rerank, total) on this hardware, with the device recorded.
  - Indexing throughput (chunks/s), peak VRAM, and wall-clock time.
- **Implementation.**
  - Metrics live in `evaluation/retrieval_metrics.py`.
  - Tests check them against `ranx` (MIT, dev dependency; version checked at M2) on fixtures.
  - Every run writes a TREC run file and per-query metrics as artifacts.

### D. Tracking (MLflow; ADR-0012)

- **One MLflow experiment per study:** `retrieval-baselines`, `reranker-finetune`, and so on.
- **Each run records:**

  | Kind | What |
  |---|---|
  | Params | RetrievalConfig YAML, model ids and HF revisions, `dataset_version`, split, seed, git SHA, device, package lock hash |
  | Metrics | per split |
  | Artifacts | run file, per-query CSV, config, and for training, the best checkpoint and its SHA-256 |

- **Reproducibility:** `uv run mezvuat-exp run experiments/<name>.yaml` must reproduce the metrics, within GPU nondeterminism, which is reported.

### E. Reranker fine-tuning protocol

- **Base model:** `cross-encoder/ms-marco-MiniLM-L6-v2`. The secondary target is gte-reranker-modernbert-base, if VRAM allows.
- **Data:**
  - (query, positive) pairs from **domain train**, optionally plus doc-disjoint SciFact train. These are two ablations.
  - Hard negatives come from `sentence_transformers.util.mine_hard_negatives` over hybrid_rrf candidates **restricted to train families**, with `relative_margin=0.05` to avoid false negatives, plus a few random negatives.
- **Training setup:**
  - `CrossEncoderTrainer` with `BinaryCrossEntropyLoss(pos_weight=num_negatives)`, the recommended strong default.
  - LambdaLoss only as an ablation, if time allows.
  - Learning rate ∈ {1e-5, 2e-5, 5e-5}, ≤ 3 epochs, warmup 10%, fp16 AMP with fp32 master weights.
  - max_length 512, batch 16–32.
- **Model selection:** `CrossEncoderRerankingEvaluator(always_rerank_positives=False)` on **val**, with `load_best_model_at_end` and early stopping.
- **Seeds:** 3. Report every seed, plus mean ± sd.
- **Comparators on test:**
  1. hybrid_rrf, no rerank.
  2. Pretrained MiniLM-L6.
  3. Pretrained bge-reranker-base, a bigger off-the-shelf model.
  4. Fine-tuned MiniLM-L6.

### F. Decision rule ("does fine-tuning win?")

- **Primary comparison:** (4) vs (2), Δ nDCG@10 on the **domain test split**.
- **Method:** paired bootstrap over queries, with 10,000 resamples, a fixed seed, and a 95% percentile CI. A paired randomization test p-value is reported alongside.
- **Fine-tuning wins only if all of the following hold:**
  - the val-selected model has a CI lower bound > 0;
  - the mean Δ over the 3 seeds is > 0;
  - **guardrail:** on SciFact test and SCIDOCS, nDCG@10 does not drop by more than 0.01 (the CI upper bound of the drop must be < 0.01);
  - rerank p95 latency is unchanged, which should hold since the architecture is the same.
- **We also report (4) vs (3).** If a larger pretrained reranker beats fine-tuning, we say so. That is a legitimate finding.
- **Power caveat.** With about 150 test queries the CI half-width will probably be around 0.03–0.05 nDCG (EST), so small gains are undetectable, and we report that rather than over-claim.
- **If the rule fails,** we publish the negative result and keep the pretrained reranker.

## Amendment, 2026-11-11: domain dataset v1 as built (decision D2)

D2 changes how val and test labels are made, from a full human second pass to product-owner review of LLM pre-labels, and caps that review at about 6 to 8 hours. The dataset files are in `data/domain/v1/`, and all prompts, models, seeds and token counts are in `manifest.json`.

**Corpus and splits.**
- 150 CC-licensed papers. The latest version of each is chunked, giving 10,125 chunks.
- The family split was committed before any query existed: 90 train, 30 val and 30 test families.

**Queries.**
- `openai/gpt-oss-20b` on Groq, one query per sampled passage, round-robin across families.
- Settings: temperature 0.7, a recorded seed per query, reasoning effort low, strict JSON schema.
- Rejection rules:
  - more than 5 consecutive words copied from the passage;
  - pointing at an unseen context ("this paper", "the authors").
- One retry per passage, then the passage is dropped.
- Near-duplicates (cosine at or above 0.95) were checked across all splits.
- Result: **val 39, test 55** queries. 6 passages were rejected; there were no duplicates.

**Pools.** Each query gets up to 8 passages:
- the source passage;
- then a round-robin by rank over the top 10 of bm25, dense, hybrid_rrf and hybrid_rrf_rerank.

That makes 752 items, at a median of 177 words each.

**Pre-labels.**
- One call per query, with that query's pooled passages in a shuffled, recorded order.
- Grades 0, 1 or 2 with a short reason; temperature 0.

**Review.**
- `mezvuat-dataset review` (guide: `docs/LABELLING_GUIDE.md`).
- Pre-labels are hidden for a seeded 20% of questions, to measure anchoring.
- Labels go to append-only files; nothing is overwritten.

**What changes from the original plan.**

| Planned | Built |
|---|---|
| test at least 150 queries, val at least 100 | test 55, val 39 |
| a 20% second-annotator overlap with Cohen's kappa | not possible with one reviewer; the anchoring measure takes its place |
| train silver labels | deferred to M9 to spare the shared Groq quota |

**Power.** Using SciFact per-query variance as a stand-in, 55 test queries give a 95% CI half-width of about 0.05 nDCG@10 on a paired difference. The minimum effect detectable with 80% power is about 0.07 to 0.09. Smaller effects must be reported as inconclusive.

**Known biases.** The rest of this ADR's limitations still apply. In addition:
- One generator writes the queries and one model writes the pre-labels, and they are the same model.
- The reviewer is one person.
- Pooling depth is shallow (8 passages per query).
- Passages outside the pool count as non-relevant, which favours the four pooled systems.

## Consequences

- Human labelling effort is the main cost, and the main blocker for M3.
- Synthetic queries stay biased toward the source chunk even with these controls. Human test labels plus pooling limit that bias but do not remove it, and the limitation is stated in reports.
- Doc-family splitting makes per-split sizes uneven. Stratification reduces that.

## Verification plan

- **Unit tests:**
  - The split function never puts two versions of one family into different splits.
  - Cross-split dedupe works.
  - The metrics match `ranx` on fixtures.
  - Bootstrap CIs on a synthetic known-difference case have the right coverage (simulation test).
- **CI check:** no file under `data/splits/test` is readable from the training entrypoint. The trainer asserts that every training `document_family_id` is in the train manifest.
- **Report:** `docs/eval/retrieval_report.md` is generated from MLflow, never written by hand.

## Sources (accessed 2026-11-08)

- huggingface.co/datasets/mteb/scifact, mteb/nfcorpus, mteb/scidocs
- github.com/allenai/scifact LICENSE.md and github.com/allenai/scidocs LICENSE
- cl.uni-heidelberg.de/statnlpgroup/nfcorpus
- github.com/beir-cellar/beir README (licenses are the user's responsibility)
- sbert.net/docs/cross_encoder/training_overview.html (losses, `mine_hard_negatives`, evaluators, overfitting guidance)
