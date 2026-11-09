# 0006. Embedding and reranker models; the four retrieval configurations; fusion

Proposed · 2026-11-08 · Spec §1.3B, §1.3C, §8 item 1

## Context

The spec asks for:
- A configurable embedder, never a hardcoded one.
- A cross-encoder reranking layer.
- A comparison of four configurations.

The models must also fit this hardware (4 GB VRAM, with CPU fallback; see ENVIRONMENT.md) and carry licenses that allow an open portfolio. This ADR picks a **shortlist and a default**. The final "best" configuration is decided by M2/M3 measurements, not here. I quote no benchmark numbers.

## Options considered

### Embedders (model cards, accessed 2026-11-08)

| Model | Params | Dim | Max len | License | Prefix | Role |
|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 22.7M | 384 | 256 (trained 128) | Apache-2.0 | none | Weak, classic baseline |
| **BAAI/bge-small-en-v1.5** | 33.4M | 384 | 512 | MIT | query: `Represent this sentence for searching relevant passages: ` | **Default** |
| intfloat/e5-small-v2 | 33.4M | 384 | 512 | MIT | `query: ` / `passage: ` (required) | Same-size alternative |
| Snowflake/snowflake-arctic-embed-s | 33.2M | 384 | 512 | Apache-2.0 | BGE-style query prefix | Same-size alternative |
| BAAI/bge-base-en-v1.5 | ~110M | 768 | 512 | MIT | as bge-small | Size step-up (GPU) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | ≤1024 (MRL) | 32k | Apache-2.0 | instruction on query | Ceiling experiment, GPU only |

Excluded models:
- **google/embeddinggemma-300m:** gated Gemma license, and its card says no fp16 activations.
- **nomic-embed-text-v1.5:** needs `trust_remote_code` on older stacks, and 8k context brings no benefit for ~350-token chunks.
- **mxbai-embed-large:** 335M parameters for the same 512-token window. Kept as an optional extra.

### Cross-encoder rerankers

| Model | Params | License | 4 GB fine-tune (EST) | Role |
|---|---|---|---|---|
| **cross-encoder/ms-marco-MiniLM-L6-v2** | 22.7M | Apache-2.0 | Easy | **Default reranker** (works on CPU) and **primary fine-tuning target** |
| BAAI/bge-reranker-base | 278M | MIT | Only with frozen embeddings | Stronger pretrained comparator (GPU inference) |
| Alibaba-NLP/gte-reranker-modernbert-base | 149M | Apache-2.0 | Tight; batch 4–8 with grad checkpointing | Secondary fine-tuning target |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | 70.8M | Apache-2.0 | Easy | Optional comparator |

Excluded rerankers:
- **jina-reranker-v2:** CC-BY-NC-4.0.
- **bge-reranker-v2-m3 and Qwen3-Reranker-0.6B:** each needs about 9 GB or more to fine-tune. They may still serve as inference-only comparators if VRAM allows; this is marked as an optional experiment.

### Fusion

| Option | Verdict |
|---|---|
| Score normalization + weighted sum (convex combination) | Needs score calibration per query and per engine, plus a tuned weight |
| Distribution-based (DBSF) | Same calibration problem |
| **Reciprocal Rank Fusion** | Uses ranks only, so it is robust to different score scales. Has one parameter. |

## Decision

- **Interfaces.** `Embedder` protocol: `embed_queries`, `embed_documents`, `dim`, `max_len`, `model_id`, `revision`. `Reranker` protocol: `score(query, passages)`.
- **Implementation.** sentence-transformers **6.2.x**, with `SentenceTransformer` for embedders and `CrossEncoder` for rerankers.
  - Models are pinned by **HF revision hash**, not just by name.
  - Prefixes are configured per model in `experiments/models.yaml`.
  - Inference runs in fp16 on CUDA and fp32 on CPU.
- **Serving default:** bge-small-en-v1.5 (dense) + bm25s (lexical) + RRF + ms-marco-MiniLM-L6-v2 (rerank). Every part is overridable per workspace and per experiment.
- **The four required configurations.** Each is a named `RetrievalConfig` and is reproducible from YAML.

  | Name | Steps |
  |---|---|
  | `dense` | Embedder → top-k |
  | `bm25` | bm25s → top-k |
  | `hybrid_rrf` | BM25 top-100 and dense top-100 → RRF → dedupe → filters → top-k |
  | `hybrid_rrf_rerank` | `hybrid_rrf` top-50 → cross-encoder → top-k |

  - Candidate depths 100 and 50 are defaults, swept on validation in M3 (rerank depth ∈ {20, 50, 100}).
  - Rerank depth trades latency against recall, and both are reported.
- **RRF:** `score(d) = Σ_i 1/(k + rank_i(d))`, with 1-based ranks and **k = 60**.
  - Why 60: Cormack, Clarke and Büttcher (SIGIR 2009) fixed k=60 in a pilot study and reported it was not critical. It is the de facto standard, which makes our numbers comparable.
  - k ∈ {10, 30, 60, 100} is swept on the **validation** split only, and k=60 stays unless another value wins with a CI that excludes zero (ADR-0007 rule).
  - Qdrant's default k=2 is **not** used.
- **Dedupe after fusion:**
  - By `chunk_id`.
  - Then near-duplicates: identical `text_sha256` across versions of the same document collapses to the latest version, unless the query is temporal (ADR-0010).
- **Embedding cost metric.** Measured on this machine:
  - chunks/s and tokens/s per device;
  - peak VRAM;
  - total wall-clock time to index the corpus;
  - USD = 0, since the models run locally.

  If a hosted embedding API is ever compared, its tokens and the provider's list price are recorded and labelled as estimates.

## Amendment, 2026-11-09: choices made on the SciFact dev split

All selection used the **dev split only** (161 queries carved from SciFact train). Test splits were evaluated once, after the choices below were fixed. Full tables with confidence intervals and MLflow run ids: `docs/eval/retrieval_report.md`, generated from `docs/eval/runs/`.

- **Embedder: `BAAI/bge-small-en-v1.5`.** Hybrid nDCG@10 on dev was 0.8025 (CI 0.754, 0.852), against 0.8034 for `bge-base-en-v1.5`, which embeds 2.5 times more slowly (272 vs 111 chunks/s on the laptop GPU) and peaked at twice the VRAM. The other small models were lower.
- **RRF k = 60 kept.** k in {10, 30, 60, 100} gave 0.8021, 0.8026, 0.8025 and 0.7995, with no measurable difference.
- **Reranker: `bge-reranker-base` at depth 50, in the rerank configuration.** On dev, every reranker failed to beat plain hybrid:

  | Reranker (depth) | nDCG@10 vs hybrid | 95% CI of the difference |
  |---|---|---|
  | ms-marco-MiniLM-L6-v2 (20) | -0.0517 | (-0.0775, -0.0254) |
  | ms-marco-MiniLM-L6-v2 (50) | -0.0572 | (-0.0860, -0.0290) |
  | ms-marco-MiniLM-L6-v2 (100) | -0.0640 | (-0.0945, -0.0341) |
  | bge-reranker-base (50) | -0.0229 | (-0.0573, +0.0102) |
  | gte-reranker-modernbert-base (50) | -0.0196 | (-0.0552, +0.0169) |

  `bge-reranker-base` and `gte-reranker-modernbert-base` are indistinguishable on dev, and the former is about 1.8 times faster (rerank p50 493 ms vs 869 ms), so it is the one carried into the rerank configuration.
- **The API default is now `hybrid_rrf`, not rerank.** This is a deviation from the original plan to rerank by default. Reranking did not help on either public benchmark and made SCIDOCS significantly worse. SciFact claims and SCIDOCS title-to-citation pairs are unlike the question-to-passage task these rerankers were trained on, so the domain dataset may behave differently; reranking stays available per request and is re-evaluated in M3 and M9. The compose image still uses MiniLM-L6 as its reranker because `bge-reranker-base` does not fit the container memory budget on this machine.

## Consequences

- Small models keep the system usable on CPU in Docker. The GPU speeds up experiments and training.
- A 384-dim default keeps Qdrant small.
- The shortlist is English-first. A multilingual corpus would need multilingual-e5 or bge-m3, which is a new ADR.

## Verification plan

- **M2:** all embedders × {dense, hybrid_rrf} on SciFact test plus SCIDOCS. Report nDCG@10, Recall@100, MRR@10, latency and indexing cost.
- **M3:** rerankers × depth on the domain validation split.
- **Default selection:** chosen on validation, confirmed once on test, and logged in MLflow with model revisions.
- **Test:** the RRF implementation matches a hand-computed example, and ties break deterministically on `chunk_id`.

## Sources (accessed 2026-11-08)

- Model cards on huggingface.co for each model listed.
- sbert.net/docs/cross_encoder/training_overview.html
- pypi.org/project/sentence-transformers (6.2.0, 2026-10-21)
- Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods", SIGIR 2009 (plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). The pilot-study wording comes from secondary summaries; I could not extract the PDF text, so the exact quote is UNVERIFIED.
