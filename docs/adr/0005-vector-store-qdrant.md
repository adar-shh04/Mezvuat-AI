# 0005. Vector store: Qdrant

Proposed · 2026-11-08 · Spec §1.5, §5.3, §8 item 2

## Context

The spec proposes Qdrant and asks us to replace it "only with a compelling technical reason". Requirements:
- Workspace isolation, with no cross-workspace retrieval (§5.3).
- Metadata filters: document, dates, latest version.
- Several embedding models side by side for experiments (§1.3C).
- The service runs in Docker on a 16 GB laptop.

## Options considered

1. **Qdrant 1.20, one collection per embedding configuration, multitenant by payload.**
2. Qdrant with one collection per workspace. That means many small HNSW graphs, and the Qdrant docs advise against it for many tenants.
3. pgvector inside Postgres. It would remove a service, but the spec already chose Qdrant. It also lacks Qdrant's tenant-optimized payload index and built-in hybrid and RRF queries. Not compelling enough to deviate.

## Decision

- **Server and client.** Qdrant server **v1.20.x** (pinned image tag) and `qdrant-client` 1.20.x.
- **Postgres is the source of truth.** Qdrant holds only rebuildable vectors plus a minimal payload, and can be rebuilt from Postgres at any time.
- **Collection naming.** One collection per embedding configuration:
  - `chunks__{embedder_slug}__{dim}__{index_version}`, for example `chunks__bge-small-en-v1.5__384__v1`.
  - Experiments create extra collections. The serving config points to one active collection, recorded in Postgres.
- **Vectors.**
  - Named vector `dense`, cosine distance, HNSW with defaults (`m=16`, `ef_construct=100`). Changes need evidence from M2 latency and recall.
  - Points stored on disk. Scalar int8 quantization is **off** until M2 shows recall stays the same.
- **Point id.** `uuid5(NAMESPACE, chunk_id)`, which is deterministic, so re-upserts are idempotent.
- **Payload.** Only filter fields; text is kept in Postgres.

  | Field | Index |
  |---|---|
  | `workspace_id` | keyword, **`is_tenant: true`** |
  | `document_id`, `document_version_id` | keyword |
  | `is_latest_version` | bool |
  | `published_at`, `effective_from`, `revised_at` | datetime, nullable |
  | `section_kind` | keyword |
  | `chunk_id`, `chunker_version` | none |

- **Mandatory tenancy.**
  - The only search entry point is `DenseRetriever.search(workspace_id: WorkspaceId, ...)`.
  - Inside `persistence/qdrant.py`, the adapter always adds a `must` condition on `workspace_id`.
  - No code path can build a query without that condition. This is enforced by the type signature and by a unit test that inspects the generated filter.
- **Fusion happens in Mezvuat, not in Qdrant's `prefetch` + RRF.** The BM25 arm lives in bm25s (ADR-0004).
  - Qdrant's RRF default is **k=2**, with zero-based ranks.
  - If Qdrant native BM25 is adopted later, server-side fusion is allowed only with an explicit `Rrf(k=60)`, and it must match our fusion on a fixed test.
- **Health.** If Qdrant is unavailable, retrieval returns `retrieval_unavailable` (HTTP 503) and the workflow stops with an explicit message. It never answers without evidence (spec §4.3).
  - BM25-only degraded mode is **not** silent. It is allowed only if the user enables it per request, and the answer is labelled "lexical only".

## Consequences

- One HNSW graph per embedder, with a tenant-aware payload index. Simple to operate.
- Experiments multiply collections, so old experiment collections need a cleanup command.
- Changing the embedder means a full re-embed into a new collection, then an atomic switch.

## Verification plan

- **Integration test:** two workspaces with overlapping content. Every query, including empty-filter edge cases, returns only the caller's workspace points.
- **Rebuild test:** drop the collection, run `reindex --workspace X`, and get identical point ids and counts.
- **Failure test:** stop the Qdrant container and confirm the API returns 503 `retrieval_unavailable` and the workflow produces no answer.

## Sources (accessed 2026-11-08)

- github.com/qdrant/qdrant/releases (v1.20.0, 2026-10-22)
- pypi.org/project/qdrant-client (1.20.0)
- qdrant.tech/documentation/guides/multiple-partitions/ (`is_tenant`)
- qdrant.tech/documentation/concepts/hybrid-queries/ (RRF, `Rrf(k)` since 1.16, default k=2)
