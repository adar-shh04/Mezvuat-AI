# Mezvuat AI

*Research intelligence. Connected knowledge. Verifiable answers.*

Mezvuat AI is a research and knowledge intelligence platform. It works over a focused document collection:
- Ingests documents with full provenance (document, version, page, section, character offsets).
- Retrieves evidence with hybrid BM25 plus dense retrieval and cross-encoder reranking.
- Runs a bounded, inspectable LangGraph research workflow.
- Answers with citations that are verified against the exact source spans, and with explicit temporal context.

Mezvuat works on its own with a direct LLM provider. It can also use [ModelMesh](../modelmesh), an independent inference gateway, through ModelMesh's versioned public API.

> **Status: Phase 2 in progress.** M0, M1, M2 and the part of M3 that needs no API key are implemented and verified: workspaces, ingestion with provenance, hybrid retrieval, and evaluation on public benchmarks. There is no LLM workflow, frontend or ModelMesh integration yet. Licence: AGPL-3.0.
>
> - `docs/spec/PRODUCT_SPEC.md`: the product specification
> - `docs/DECISIONS.md`: product owner decisions, which override the ADRs
> - `docs/ENVIRONMENT.md`: the verified development environment
> - `docs/adr/`: architecture decision records
> - `docs/eval/retrieval_report.md`: measured retrieval results, generated from run files
> - `docs/contract/`: mirror of the ModelMesh inference contract
> - `docs/IMPLEMENTATION_PLAN.md`: milestones M0 to M9

## Run it

```
docker compose -f infra/compose.yaml up -d --build
python scripts/smoke.py <a-paper.pdf> "<a query>"
```

Ports are bound to 127.0.0.1: API 18000, Postgres 15433, Qdrant 16333. Backend development needs `uv`; see `scripts/dev-env.sh`.
