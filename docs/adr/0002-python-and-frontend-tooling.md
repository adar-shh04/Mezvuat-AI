# 0002. Python, backend and frontend tooling

Proposed · 2026-11-08 · Spec §1.5, §5.2

## Context

The machine has only Python 3.14.3 and no uv (see ENVIRONMENT.md). The stack needs:
- PyTorch with CUDA.
- sentence-transformers.
- LangGraph.
- RAGAS, which depends on scikit-network.
- Qdrant client, FastAPI and Pydantic v2.
- A Next.js frontend.

## Options considered

**Python version:**
- **3.14:** blocked. scikit-network has no cp314 wheel.
- **3.13:** every wheel we need exists, including torch cu132, scikit-network, psycopg-binary and pyarrow.
- **3.12:** also works, but it gets security fixes only.

**Environment manager:**
- **uv:** a lockfile, a managed interpreter, and a PyTorch index per platform.
- **pip + venv:** no lockfile.
- **Poetry:** slower, and weaker PyTorch index handling.

**Type checker:**
- **mypy with the pydantic plugin:** stable and pydantic-aware.
- **pyright:** strong too, but brings a Node runtime into the Python CI.
- **ty:** still beta (0.0.99, "breaking changes can happen between any two versions").

**Frontend package manager:**
- **npm:** already installed.
- **pnpm:** not installed. Corepack is unbundled from Node 25 on.

## Decision

| Concern | Choice (version checked 2026-11-08) | Why |
|---|---|---|
| Interpreter | **CPython 3.13**, installed with `uv python install 3.13` and pinned in `.python-version` | Newest version with the full set of Windows wheels (scikit-network 0.33.5 has a cp313 `win_amd64` wheel; checked on PyPI). |
| Env and lock | **uv 0.13.x**, committed `uv.lock`, dependency groups `ml`, `eval`, `dev` | Reproducibility, which spec §5.2 requires for ML experiments. |
| PyTorch | **torch 2.15.x**. Windows/Linux GPU builds come from the `https://download.pytorch.org/whl/cu132` index, declared as `[[tool.uv.index]] explicit = true` and pinned through `tool.uv.sources`. The Docker image uses the CPU index. | cu126 and cu130 are no longer built for 2.15. Ampere is still supported. |
| Web framework | **FastAPI 0.16x**, **Pydantic 2.14**, **pydantic-settings** | Named by the spec. Gives typed, validated contracts. |
| DB access | **SQLAlchemy 2.1** (async) + **psycopg 3.3** + **Alembic** | Spec §5.2 requires migrations. psycopg 3 is also what `langgraph-checkpoint-postgres` uses. |
| HTTP client | **httpx** | Used by the ModelMesh and Groq clients (ADR-0011) and in tests. |
| Lint/format | **ruff 0.17.x** (lint + format) | One tool, fast. |
| Types | **mypy 2.x** with the pydantic plugin, `strict` on `core`, `inference`, `citations` and `retrieval` | Stable and pydantic-aware. ty can run later as a non-blocking job. |
| Tests | **pytest 9.x**, pytest-asyncio, respx (httpx mocking). Integration tests run against the compose Postgres/Qdrant. | |
| Logging | stdlib `logging` with a JSON formatter (structlog is optional; decided in M1, only if it pays off) | Fewer dependencies. |
| Frontend | **Next.js 16.4.x** (App Router, Turbopack default), **React 19**, **TypeScript**, **npm**, Node 24 LTS | Named by the spec. Node 24 meets Next's requirement of ≥ 20.9. |
| Frontend extras | Tailwind CSS. Generated API types via `openapi-typescript` from FastAPI's OpenAPI. No state library at the start. | Keeps the backend and frontend contracts typed end to end. |

Each dependency is justified when it is added. "Popular" is not a justification (spec §8).

## Consequences

- The user must install uv and let it download CPython 3.13. Both happen inside the user profile, and nothing heavy is installed globally.
- The mypy plugin adds a little configuration work.
- The torch CUDA wheel is large (≥ 2 GB, EST), and it lives only in the local `.venv` on X:.
- Two lock files: `uv.lock` and `package-lock.json`.

## Verification plan

- **M1 smoke test:**
  1. `uv sync --group dev` on a clean clone.
  2. `uv run ruff check`, `uv run mypy`, `uv run pytest` all pass.
  3. `npm ci && npm run build` in `frontend/` passes.
- **M2 GPU check:** `uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_arch_list())"` prints `sm_86`, or a compatible PTX target.
- **CI** runs on `ubuntu-latest` with Python 3.13 and CPU torch, so it needs no GPU.

## Sources (accessed 2026-11-08)

- PyPI JSON: scikit-network, ragas, langgraph, uv, torch.
- `https://download.pytorch.org/whl/cu132/torch/`, which lists `torch-2.15.0+cu132-cp313-cp313-win_amd64.whl`.
- PyTorch 2.15 release notes: github.com/pytorch/pytorch/releases/tag/v2.15.0
- uv docs: docs.astral.sh/uv/guides/integration/pytorch/ and docs.astral.sh/uv/concepts/python-versions/
- ty status: pypi.org/pypi/ty
- Next.js: registry.npmjs.org/next/latest (16.4.5) and nextjs.org/docs/app/getting-started/installation
