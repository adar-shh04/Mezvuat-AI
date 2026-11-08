# 0004. Lexical retrieval: BM25 engine

Accepted with evidence · 2026-11-09 (proposed 2026-11-08) · Spec §1.3B, §1.3C, §8 item 3

## Context

The spec names **BM25** as the lexical method and lists "Postgres FTS initially or a dedicated BM25 implementation". Several requirements follow from that:
- The BM25 arm is compared against dense, hybrid and reranked arms on public benchmarks. Its scoring has to be **real, documented BM25**, so a baseline number means what it says.
- Lexical search must respect workspace isolation and metadata filters, the same as dense search.
- Corpus size is small: up to about 20k chunks per workspace and about 50k in total for the demo.

## Options considered

| Option | True BM25? | License | Ops cost | Notes (checked 2026-11-08) |
|---|---|---|---|---|
| PostgreSQL `ts_rank` / `ts_rank_cd` | **No.** The docs say the ranking functions "do not use any global information": no IDF, no average document length | PostgreSQL | None | Fine for a UI title search. Not credible as a BM25 baseline. Rejected for ranking. |
| ParadeDB `pg_search` 0.26 (Tantivy) | Yes | **AGPL-3.0** (community) | Custom Postgres image | AGPL, plus a non-standard Postgres image. Rejected. |
| VectorChord-bm25 | Yes | AGPL-3.0 or ELv2 | Custom image; "tested against English" only | Rejected on license. |
| Timescale `pg_textsearch` 1.5 | Yes (k1=1.2, b=0.75, BMW) | PostgreSQL License | Needs an extension build on PG 17/18. Official Docker image UNVERIFIED | Promising, but has no verified image and statistics are per partition. Deferred. |
| Qdrant sparse vectors + FastEmbed `Qdrant/bm25` + `modifier: idf` | Approximate | Apache-2.0 | None, since Qdrant is already in the stack | The client computes TF normalization with **avg_len fixed at 256 by default**, not the corpus mean. Fixing that means re-embedding. IDF is computed per shard. |
| Qdrant **v1.20 native BM25 on the full-text payload index** | Yes; the server computes avgdl, and k1/b are set per query | Apache-2.0 | None | Released 2026-10-22, so 2 weeks old. Tokenizer, stemming and isolation details are UNVERIFIED. Strong candidate once proven. |
| **bm25s 0.3.16** (in-process, Python) | Yes. Default variant is Lucene; Robertson, ATIRE, BM25L and BM25+ are also available | MIT | An index file per workspace | Fast. README reports ~950 q/s on SciFact single-threaded. Reproducible and widely used for BEIR baselines. Rebuild cost is negligible at our scale. |
| rank_bm25 0.2.2 | Yes | Apache-2.0 | | Unmaintained since 2022. Rejected. |

## Decision

1. **Interface.** `LexicalRetriever.search(workspace_id, query, k, filters) -> list[(chunk_id, score)]`. Fusion only ever sees ranks, so engines can be swapped.
2. **Default engine: bm25s.**
   - Parameters: Lucene variant, k1=1.2, b=0.75.
   - Text processing: English stopwords and PyStemmer (Snowball English).
   - These parameters are part of `retriever_config` and are logged to MLflow.
3. **Index layout.**
   - **One index per `(workspace_id, index_version)`**, stored on the data volume as `bm25/{workspace}/{index_version}/`, with an array mapping row number to chunk_id.
   - Isolation is **physical**: a query can only load its own workspace's index.
   - Ingestion commits rebuild the workspace index in the worker. With ≤ 20k chunks, a rebuild takes seconds (EST).
   - The new index is written to a temporary directory, then a pointer in Postgres (`workspace.lexical_index_version`) is swapped atomically.
   - The API reloads lazily when it sees a new version (an LRU of loaded indices).
4. **Metadata filters.**
   - Filters (document ids, date ranges, latest version only) resolve in Postgres to an allowed set of chunk row numbers.
   - That set is applied as a mask before top-k. If bm25s weight masking is unavailable, we over-fetch by 5× and filter after scoring; which path we use is checked in M2.
5. **Postgres FTS** is used only for document title and metadata search in the UI. It is never part of the ranking pipeline.
6. **M2 spike: Qdrant v1.20 native BM25 vs bm25s.**
   - Run both on SciFact and the domain validation split.
   - **Adoption rule:** if nDCG@10 differs by less than 0.005 with overlapping 95% bootstrap CIs, and p95 latency is ≤ 50 ms, and per-workspace statistics are confirmed, we switch to Qdrant native BM25. That removes the file-index lifecycle.
   - Otherwise bm25s stays.
   - The result is recorded as an amendment to this ADR.

## Amendment, 2026-11-09: decided with evidence (decisions D6 and D7)

D6 allows AGPL dependencies but requires this choice to be made with evidence and to keep the simpler option unless the difference is measurable. All numbers are from `experiments/spikes/bm25_engines.py`, on the SciFact dev split (161 queries carved from train; the test split was not touched). Raw results and CIs: `docs/eval/bm25_engines.json`.

| Engine | nDCG@10 (95% CI) | Recall@100 | MRR@10 | Index build | Query p50 / p95 |
|---|---|---|---|---|---|
| **bm25s** 0.3.16 (Lucene variant) | 0.7466 (0.689, 0.804) | 0.941 | 0.7206 | 1.17 s | 1.43 / 2.59 ms |
| ParadeDB pg_search 0.26.6 | 0.7442 (0.686, 0.802) | 0.941 | 0.7177 | 0.48 s | 3.53 / 5.38 ms |
| Qdrant 1.20 native BM25 | 0.7569 (0.701, 0.812) | 0.941 | 0.7287 | 1.08 s | 5.09 / 25.72 ms |

Paired against bm25s (bootstrap over queries, 10,000 resamples):

| Pair | Delta nDCG@10 | 95% CI | Randomisation p |
|---|---|---|---|
| ParadeDB - bm25s | -0.0024 | (-0.0055, 0.0000) | 0.107 |
| Qdrant native - bm25s | +0.0103 | (-0.0048, +0.0273) | 0.219 |

Neither difference is significant. Recall@100 was identical for all three.

**Operational cost.** ParadeDB needs its own 1.53 GB Postgres image in place of the stock `postgres:17-alpine`, and it would make the AGPL extension part of the database. Its build is faster here, which does not matter at this corpus size.

**Decision: keep bm25s.** ParadeDB shows no measurable gain for a heavier, non-standard database image. Qdrant native BM25 does not meet the adoption rule in point 6 (absolute delta under 0.005, per-workspace statistics confirmed): the delta is 0.0103, with a CI that includes zero, and per-workspace IDF behaviour was not tested. It remains the most likely future replacement, since it would remove the file-index lifecycle. Worth re-testing on the domain dataset.

**Metadata filters** use bm25s `weight_mask`, so filtering happens before top-k and the over-fetch fallback in point 4 was not needed.

**Limitations.** One public dataset of short abstracts, 161 queries; differences smaller than about 0.02 would not be detectable here. Tokenisers differ slightly between engines.

## Consequences

- A credible, documented BM25 baseline from day one, with no AGPL code and no custom Postgres image.
- Costs: index files to manage, an index-pointer consistency step, and per-process memory for loaded indices. All are small at our scale.
- **Does not scale to millions of chunks per workspace.** That is out of scope (spec §1.2). The interface allows migrating to Qdrant BM25 or pg_textsearch later.

## Verification plan

- **Unit test** against a hand-computed BM25 score for a 3-document toy corpus, for the Lucene formula.
- **Sanity check** on SciFact: our BM25 nDCG@10 must land within ±0.02 of the BM25 figure that the bm25s or BEIR repos publish. We record the source and do not quote it here until measured. A large gap means a tokenization bug.
- **Isolation test:** a query in workspace A can never return a chunk from workspace B, including after concurrent re-indexing.
- **Latency:** p50/p95 lexical latency at 20k chunks, logged.

## Sources (accessed 2026-11-08)

- PostgreSQL 18 docs, textsearch-controls ("do not use any global information").
- github.com/paradedb/paradedb
- github.com/timescale/pg_textsearch
- github.com/tensorchord/VectorChord-bm25
- fastembed `sparse/bm25.py` (avg_len=256)
- qdrant.tech/documentation/concepts/indexing/
- github.com/qdrant/qdrant/releases (v1.20.0: "BM25 scoring on the full-text payload index"; v1.19.0: per-query IDF corpus)
- github.com/xhluca/bm25s and pypi.org/project/bm25s
