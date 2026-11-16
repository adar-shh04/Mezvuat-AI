# Combined benchmark: direct vs ModelMesh routing policies (M8)

**Status: not yet run.** There are no numbers here on purpose (acceptance C7: the report is generated from data,
never typed in). The script exists and its plumbing is tested with fakes only
(`backend/tests/unit/test_combined_benchmark.py`).

## Why it has not run

- ModelMesh is busy with its own benchmark this week, and Mezvuat must not send it requests until live integration.
- The eval set needs the owner's reviewed gold items (decision D11). Drafts are skipped by `read_items`.

## How to run it, once both are ready

From `backend/`, in Git Bash, after `. ../scripts/dev-env.sh`:

```bash
docker compose -f ../infra/compose.yaml up -d postgres qdrant
# .env: MODELMESH_BASE_URL, MODELMESH_API_KEY, MEZVUAT_SECRET_KEY and GROQ_API_KEY set; a public-corpus workspace
uv run --no-sync mezvuat-exp combined-benchmark \
  --items ../data/domain/temporal/v1/<reviewed gold items>.jsonl \
  --workspace <public workspace id> \
  --arms direct,modelmesh:cost_optimized,modelmesh:quality_balanced,modelmesh:latency_optimized \
  --out ../docs/eval/combined_benchmark
docker compose -f ../infra/compose.yaml stop postgres qdrant
```

This overwrites this file with the generated report and writes `results.json` (every row, with run ids and `mmr_`
ids, so run ↔ ModelMesh request ↔ feedback can be joined). One MLflow run per arm is logged unless `--no-mlflow`.

## What it measures

- Every item through every arm, items in the outer loop so an outage affects all arms alike.
- Quality: Mezvuat's per-run metrics (completed, abstention correctness against `expected_behavior`, cited versions
  against `expected_versions`, supported-claim rate). M6's RAGAS and custom metrics plug in through `metrics=`.
- Cost and latency for ModelMesh arms: `GET /v1/requests/{id}` for every call (estimated cost including failed
  attempts, and `latency_ms.total`). Direct arms: Mezvuat's own call timing, cost "not reported".
- ModelMesh arms run with `INFERENCE_MODE=modelmesh` semantics (no fallback), so a direct answer never counts as a
  ModelMesh one.
