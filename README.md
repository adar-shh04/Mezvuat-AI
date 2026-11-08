# Mezvuat AI

*Research intelligence. Connected knowledge. Verifiable answers.*

Mezvuat AI is a research and knowledge intelligence platform. It works over a focused document collection:
- Ingests documents with full provenance (document, version, page, section, character offsets).
- Retrieves evidence with hybrid BM25 plus dense retrieval and cross-encoder reranking.
- Runs a bounded, inspectable LangGraph research workflow.
- Answers with citations that are verified against the exact source spans, and with explicit temporal context.

Mezvuat works on its own with a direct LLM provider. It can also use [ModelMesh](../modelmesh), an independent inference gateway, through ModelMesh's versioned public API.

> **Status: architecture phase.** No application code has been written yet. See:
> - `docs/spec/PRODUCT_SPEC.md`: the product specification (source of truth)
> - `docs/ENVIRONMENT.md`: the verified development environment and its constraints
> - `docs/adr/`: architecture decision records (Proposed)
> - `docs/contract/`: a mirror of the ModelMesh inference contract, plus Mezvuat's change requests
> - `docs/IMPLEMENTATION_PLAN.md`: the milestones M0–M9

Implementation starts only after the product owner authorizes it.
