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
| DB access | **SQLAlchemy 2.1** (sync, amended in Phase 2) + **psycopg 3.3** + **Alembic** | Spec §5.2 requires migrations. psycopg 3 is also what `langgraph-checkpoint-postgres` uses. Async was dropped: psycopg async cannot run on the Windows ProactorEventLoop, and embedding and reranking block a thread anyway. FastAPI runs sync routes in its threadpool. |
| HTTP client | **httpx** | Used by the ModelMesh and Groq clients (ADR-0011) and in tests. |
| Lint/format | **ruff 0.17.x** (lint + format) | One tool, fast. |
| Types | **mypy 2.x** with the pydantic plugin, `strict` on `core`, `inference`, `citations` and `retrieval` | Stable and pydantic-aware. ty can run later as a non-blocking job. |
| Tests | **pytest 9.x**, pytest-asyncio, respx (httpx mocking). Integration tests run against the compose Postgres/Qdrant. | |
| Logging | stdlib `logging` with a JSON formatter (structlog is optional; decided in M1, only if it pays off) | Fewer dependencies. |
| Frontend | **Next.js 16.4.x** (App Router, Turbopack default), **React 19**, **TypeScript**, **npm**, Node 24 LTS | Named by the spec. Node 24 meets Next's requirement of ≥ 20.9. |
| Frontend extras | Tailwind CSS. Generated API types via `openapi-typescript` from FastAPI's OpenAPI. No state library at the start. | Keeps the backend and frontend contracts typed end to end. |

Each dependency is justified when it is added. "Popular" is not a justification (spec §8).

## Phase 2 notes (2026-11-09)

- Python is CPython 3.13.18 through uv; CUDA works with torch 2.15.0+cu132 (see `docs/ENVIRONMENT.md`).
- `ruff` line length is 120.
- Dependency groups are `ml`, `eval` and `dev` (PEP 735). The API image installs only `ml`.

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

## Amendment, 2026-11-15: frontend as built in M7

**Versions, checked on the npm registry on 2026-11-15.**

| Package | Version | Why this one |
|---|---|---|
| `next` | **16.4.6** | `latest` dist-tag; 16 is still the current stable major (17 exists only as canary builds of 16.5). The plan's "Next.js 16" holds. Requires Node ≥ 20.9; we use Node 24.13 LTS locally, in CI and in the image. |
| `react`, `react-dom` | 19.3.1 | `latest`; Next 16 accepts `^19`. |
| `typescript` | **5.9.3**, not 7.0.2 | 7.x is the native port. `openapi-typescript` 7.13 declares `typescript: ^5.x` and `typescript-eslint` 8.72 declares `<6.1.0`, so 5.9 is the newest version every tool supports. |
| `eslint` | **9.39.5**, not 10.x | `eslint-config-next` 16.4.6 pulls in `eslint-plugin-react`, `-import` and `-jsx-a11y`, whose peer ranges stop at ESLint 9. npm marks 9.x as no longer supported; move to 10 when those plugins do. |
| `react-markdown` 10.1.0 + `rehype-sanitize` 6.0.0 | runtime | Sanitized Markdown (ADR-0009 §5) without `dangerouslySetInnerHTML`: raw HTML is skipped, the tree is sanitized with GitHub's schema, unsafe URLs are dropped, images are not rendered and links show their URL. Both MIT, from the unified collective. |
| `openapi-typescript` 7.13.0 | dev | Generates `src/lib/api-types.ts` from `frontend/openapi.json` (written by `scripts/export_openapi.py`). Types only, no runtime client; the fetch wrapper is 100 lines in `src/lib/api.ts`. |
| `@playwright/test` 1.65.1 | dev | See below. |

**Deviations from the decision table above.**
- **No Tailwind.** One hand-written `globals.css` (about 400 lines) covers the UI. Tailwind 4 would add `tailwindcss`, `@tailwindcss/postcss` and a native binary for a handful of screens. Revisit if the UI grows.
- **No state or data-fetching library**, as planned. React state plus `EventSource` is enough.
- **A same-origin proxy** (`src/app/api/v1/[...path]/route.ts`) forwards `/api/v1/*` to `MEZVUAT_API_URL` and adds `MEZVUAT_API_KEY` on the server. The key never reaches the browser, there is no CORS configuration, and uploads and SSE stream through unbuffered.
- **`agentRules: false`** in `next.config.ts`: `next dev` otherwise writes an `AGENTS.md` into the repository.
- **`output: "standalone"` only in the Docker build** (`NEXT_OUTPUT=standalone`), because `next start` does not support standalone output and CI tests the production build with `next start`.

**Why Playwright is a dev dependency.** The M7 acceptance test is behavioural: clicking a citation must open the right document, page and highlighted span. That needs a real browser running the real `EventSource`, Markdown sanitizer and DOM, which a unit-test runner such as Vitest with jsdom cannot give. Playwright (Apache-2.0) runs Chromium only: locally the headless shell (`npx playwright install --only-shell chromium`, about 290 MB under `PLAYWRIGHT_BROWSERS_PATH` on X:), in CI with `--with-deps`. It is a dev dependency, so it never enters the image. It also serves the stand-in API through `page.route`, so no second mocking library is needed.

**Licences.** `frontend/scripts/check-licenses.mjs` checks every installed npm package; the result is in `docs/FRONTEND_DEPENDENCY_LICENSES.md`. On 2026-11-15: 452 packages (112 runtime), all MIT, ISC, BSD, Apache-2.0, 0BSD, BlueOak, CC0, CC-BY-4.0 (`caniuse-lite` data), Python-2.0, MPL-2.0, or Apache-2.0 AND LGPL-3.0-or-later (`sharp` and its libvips binaries, an optional dependency of `next`). All are compatible with AGPL-3.0.

**Caches.** `scripts/dev-env.sh` sets `npm_config_cache` and `PLAYWRIGHT_BROWSERS_PATH` under `X:/mezvuat-cache` and `NEXT_TELEMETRY_DISABLED=1`.
