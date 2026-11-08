# Architecture Decision Records

- **Status values:** `Proposed`, then `Accepted` once the product owner approves, then `Superseded by NNNN`.
- **Current state:** every ADR is **Proposed** (architecture phase, 2026-11-08).
- **Facts:** version and licence facts were checked on 2026-11-08 and cite their sources. Where a fact is unverified, the ADR says so.

## Template

```
# NNNN. Title
Status · Date · Spec refs
## Context
## Options considered
## Decision
## Consequences
## Verification plan
## Sources
```

## Index

| # | Title | Spec |
|---|---|---|
| [0001](0001-repository-layout-and-module-boundaries.md) | Repository layout and module boundaries | §1.5, §5.1 |
| [0002](0002-python-and-frontend-tooling.md) | Python, backend and frontend tooling | §1.5, §5.2 |
| [0003](0003-document-parsing-storage-and-chunking.md) | Document parsing, storage, dedupe and chunking | §1.3A, §8.9 |
| [0004](0004-lexical-retrieval-bm25.md) | Lexical retrieval: BM25 engine | §1.3B, §8.3 |
| [0005](0005-vector-store-qdrant.md) | Vector store: Qdrant | §8.2, §5.3 |
| [0006](0006-embedding-and-reranker-models.md) | Embedding and reranker models, fusion | §1.3C, §8.1 |
| [0007](0007-retrieval-evaluation-and-reranker-fine-tuning.md) | Retrieval eval dataset and fine-tuning protocol | §1.3C, §3.4 |
| [0008](0008-langgraph-research-workflow.md) | LangGraph research workflow | §1.3D, §3.3 |
| [0009](0009-citations-verification-and-injection-defense.md) | Citations, verification, prompt-injection defence | §1.3F, §5.3 |
| [0010](0010-temporal-metadata-model.md) | Temporal metadata model | §1.3E |
| [0011](0011-inference-layer.md) | Inference layer: ModelMesh client and direct-provider fallback | §1.5, §4.3, §8.7 |
| [0012](0012-evaluation-and-observability.md) | Evaluation and observability | §1.3G, §8.8 |
| [0013](0013-demo-corpus.md) | Demo corpus and domain | §1.2 |
| [0014](0014-deployment.md) | Deployment | §8.10 |
