"""The combined benchmark (M8, acceptance C7): one eval set, run through the direct backend and through ModelMesh under
two or three routing policies.

- Quality comes from Mezvuat's own measurements of each run (below), never from ModelMesh.
- Cost and latency for ModelMesh arms come from `GET /v1/requests/{mmr id}` for every call the run made.
- Direct arms report latency from Mezvuat's own `llm_call` timing; cost is "not reported" (the direct backends give
  none, and nothing is estimated).
- ModelMesh arms run with no fallback: a call ModelMesh cannot serve fails the item instead of silently becoming a
  direct call, so each arm measures what it says.
- Every number in the report is computed from the per-item JSON written next to it. Nothing is typed in by hand.

Decision D1: the workspace must be a verified public-corpus workspace, because the direct arm uses Groq.

Per-item metrics (M6 adds RAGAS and the custom metrics; they plug in through `metrics=`):
- `completed`: the run finished without a fatal error.
- `abstention_correct`: the run abstained exactly when the gold item expects it (`expected_behavior`).
- `version_match`: every cited version is one of the gold `expected_versions` (items with expected versions only).
- `supported_claim_rate`: supported claims over all claims the run verified.

Usage (from backend/, with the stack up and ModelMesh reachable; NOT RUN YET):
    uv run --no-sync mezvuat-exp combined-benchmark --items ITEMS.jsonl --workspace ws_... \
        --arms direct,modelmesh:cost_optimized,modelmesh:quality_balanced,modelmesh:latency_optimized \
        --out ../docs/eval/combined_benchmark
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from mezvuat.core.ids import new_id
from mezvuat.orchestration.state import ResearchState

log = logging.getLogger(__name__)

REPORT_VERSION = "combined-benchmark-v1"
POLICIES = ("cost_optimized", "latency_optimized", "quality_balanced", "quality_max", "reliability_first")


@dataclass(frozen=True)
class Arm:
    name: str
    backend: str  # "direct" | "modelmesh"
    policy: str | None = None

    @classmethod
    def parse(cls, spec: str) -> Arm:
        if spec == "direct":
            return cls("direct", "direct")
        backend, _, policy = spec.partition(":")
        if backend != "modelmesh" or policy not in POLICIES:
            raise ValueError(f"unknown arm {spec!r}: use direct or modelmesh:<policy> with a policy from {POLICIES}")
        return cls(f"modelmesh:{policy}", "modelmesh", policy)


ItemMetrics = Callable[[Mapping[str, Any], ResearchState], dict[str, float]]
RunItem = Callable[[Arm, Mapping[str, Any], str], ResearchState]
FetchRecord = Callable[[str], Mapping[str, Any] | None]


def abstained(state: ResearchState) -> bool:
    answer = state.answer
    return answer is None or not any(claim.status == "supported" for claim in answer.claims)


def default_metrics(item: Mapping[str, Any], state: ResearchState) -> dict[str, float]:
    out: dict[str, float] = {"completed": float(state.status == "completed")}
    expected = item.get("expected_behavior")
    if expected is not None:
        out["abstention_correct"] = float(abstained(state) == (expected != "answer"))
    answer = state.answer
    if answer is not None and answer.claims:
        out["supported_claim_rate"] = sum(c.status == "supported" for c in answer.claims) / len(answer.claims)
    expected_versions = item.get("expected_versions")
    if expected_versions and answer is not None and answer.citations:
        cited = {c.version_label for c in answer.citations}
        out["version_match"] = float(cited <= set(expected_versions))
    return out


def _cost_and_latency(arm: Arm, state: ResearchState, fetch: FetchRecord) -> dict[str, Any]:
    """Per-run cost and LLM latency. ModelMesh arms read the execution records; direct arms use Mezvuat's timing."""
    if arm.backend == "direct":
        return {
            "llm_latency_ms": sum(c.latency_ms for c in state.llm_calls),
            "cost_estimated_usd": None,
            "records_missing": 0,
        }
    cost = 0.0
    latency = 0
    missing = 0
    for call in state.llm_calls:
        if not call.modelmesh_request_id:
            missing += 1
            continue
        record = fetch(call.modelmesh_request_id)
        meta = record.get("modelmesh") if record else None
        if not isinstance(meta, Mapping):
            missing += 1
            continue
        costs = meta.get("cost") or {}
        cost += float(costs.get("estimated_usd") or 0.0) + float(costs.get("failed_attempts_estimated_usd") or 0.0)
        latency += int((meta.get("latency_ms") or {}).get("total") or 0)
    return {"llm_latency_ms": latency, "cost_estimated_usd": cost, "records_missing": missing}


@dataclass
class BenchmarkResult:
    started_at: str
    arms: list[Arm]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        per_arm: dict[str, Any] = {}
        for arm in self.arms:
            rows = [r for r in self.rows if r["arm"] == arm.name]
            metrics: dict[str, dict[str, float | int]] = {}
            names = sorted({k for r in rows for k in r["metrics"]})
            for name in names:
                values = [r["metrics"][name] for r in rows if name in r["metrics"]]
                metrics[name] = {"mean": statistics.fmean(values), "n": len(values)}
            costs = [r["cost_estimated_usd"] for r in rows if r["cost_estimated_usd"] is not None]
            latencies = [r["llm_latency_ms"] for r in rows]
            walls = [r["wall_ms"] for r in rows]
            per_arm[arm.name] = {
                "items": len(rows),
                "metrics": metrics,
                "cost_estimated_usd_total": sum(costs) if costs else None,
                "llm_latency_ms_median": statistics.median(latencies) if latencies else None,
                "wall_ms_median": statistics.median(walls) if walls else None,
                "records_missing": sum(r["records_missing"] for r in rows),
                "errors": sorted({r["error"] for r in rows if r["error"]}),
            }
        return {
            "report_version": REPORT_VERSION,
            "started_at": self.started_at,
            "arms": [arm.name for arm in self.arms],
            "per_arm": per_arm,
        }


def run_benchmark(
    items: Sequence[Mapping[str, Any]],
    arms: Sequence[Arm],
    run_item: RunItem,
    fetch_record: FetchRecord,
    *,
    metrics: ItemMetrics = default_metrics,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Every item through every arm, items in the outer loop so a time-dependent outage hits all arms alike."""
    result = BenchmarkResult(started_at=datetime.now(UTC).isoformat(), arms=list(arms))
    for item in items:
        for arm in arms:
            run_id = new_id("run")
            started = clock()
            state = run_item(arm, item, run_id)
            wall_ms = int((clock() - started) * 1000)
            accounting = _cost_and_latency(arm, state, fetch_record)
            result.rows.append(
                {
                    "item_id": item["item_id"],
                    "arm": arm.name,
                    "run_id": run_id,
                    "status": state.status,
                    "error": state.error.code if state.error else None,
                    "metrics": metrics(item, state),
                    "wall_ms": wall_ms,
                    "llm_calls": len(state.llm_calls),
                    "modelmesh_request_ids": [
                        c.modelmesh_request_id for c in state.llm_calls if c.modelmesh_request_id
                    ],
                    "fallback_calls": sum(1 for c in state.llm_calls if c.fallback_reason),
                    **accounting,
                }
            )
            log.info("benchmark item done", extra={"item": item["item_id"], "arm": arm.name, "status": state.status})
    return result


def render_markdown(summary: Mapping[str, Any], *, mlflow_runs: Mapping[str, str] | None = None) -> str:
    """The report, generated only from `summary` (C7)."""
    lines = [
        "# Combined benchmark: direct vs ModelMesh routing policies (M8)",
        "",
        f"Generated from `results.json` by `{REPORT_VERSION}`; started {summary['started_at']}.",
        "Quality: Mezvuat's per-run metrics. Cost and latency for ModelMesh arms: `GET /v1/requests/{id}`; cost is",
        "ModelMesh's estimate, not billing data. Direct arms report no cost.",
        "",
    ]
    names = sorted({m for arm in summary["per_arm"].values() for m in arm["metrics"]})
    header = ["arm", "items", *names, "est. cost (USD)", "median LLM latency (ms)", "median wall (ms)", "errors"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm_name in summary["arms"]:
        arm = summary["per_arm"][arm_name]
        cells = [arm_name, str(arm["items"])]
        for name in names:
            metric = arm["metrics"].get(name)
            cells.append(f"{metric['mean']:.3f} (n={metric['n']})" if metric else "n/a")
        cost = arm["cost_estimated_usd_total"]
        cells.append("not reported" if cost is None else f"{cost:.5f}")
        cells.append("n/a" if arm["llm_latency_ms_median"] is None else f"{arm['llm_latency_ms_median']:.0f}")
        cells.append("n/a" if arm["wall_ms_median"] is None else f"{arm['wall_ms_median']:.0f}")
        cells.append(", ".join(arm["errors"]) or "none")
        lines.append("| " + " | ".join(cells) + " |")
    if mlflow_runs:
        lines += ["", "MLflow runs: " + ", ".join(f"{arm} `{run}`" for arm, run in sorted(mlflow_runs.items()))]
    missing = sum(arm["records_missing"] for arm in summary["per_arm"].values())
    if missing:
        lines += [
            "",
            f"{missing} ModelMesh calls had no readable execution record; their cost and latency are missing.",
        ]
    return "\n".join(lines) + "\n"


def log_to_mlflow(summary: Mapping[str, Any], *, experiment: str = "combined-benchmark") -> dict[str, str]:
    """One MLflow run per arm. Returns arm -> MLflow run id. Requires the `eval` dependency group."""
    import mlflow

    mlflow.set_experiment(experiment)
    runs: dict[str, str] = {}
    for arm_name, arm in summary["per_arm"].items():
        with mlflow.start_run(run_name=arm_name) as run:
            mlflow.log_params({"arm": arm_name, "report_version": summary["report_version"], "items": arm["items"]})
            values = {f"{k}_mean": v["mean"] for k, v in arm["metrics"].items()}
            if arm["cost_estimated_usd_total"] is not None:
                values["cost_estimated_usd_total"] = arm["cost_estimated_usd_total"]
            if arm["llm_latency_ms_median"] is not None:
                values["llm_latency_ms_median"] = float(arm["llm_latency_ms_median"])
            mlflow.log_metrics(values)
            runs[arm_name] = run.info.run_id
    return runs


def read_items(path: Path, *, reviewed_only: bool = True) -> list[dict[str, Any]]:
    """Eval items, one JSON object per line with at least `item_id` and `question`. With `reviewed_only`, items whose
    `status` says they are unreviewed drafts are skipped: only gold items are benchmarked (D11)."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if reviewed_only and item.get("status") in ("draft", "rejected"):
            continue
        items.append(item)
    return items


def write_outputs(out_dir: Path, result: BenchmarkResult, mlflow_runs: Mapping[str, str] | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    payload = {"summary": summary, "rows": result.rows, "mlflow_runs": dict(mlflow_runs or {})}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    report = out_dir / "report.md"
    report.write_text(render_markdown(summary, mlflow_runs=mlflow_runs), encoding="utf-8", newline="\n")
    return report


def production_runner(workspace_id: str, arms: Iterable[Arm]) -> tuple[RunItem, FetchRecord]:
    """The real stack: the API's retrieval pipeline and workflow, with one LLM client per arm."""
    import dataclasses

    from mezvuat.api.deps import production_workflow
    from mezvuat.core.config import get_settings
    from mezvuat.inference import modelmesh_ops
    from mezvuat.inference.factory import direct_client, modelmesh_client
    from mezvuat.inference.modelmesh import ModelMeshClient
    from mezvuat.orchestration import Budget, build_graph, run_to_end
    from mezvuat.persistence.db import get_session_factory
    from mezvuat.persistence.models import Workspace

    settings = get_settings()
    with get_session_factory()() as session:
        workspace = session.get(Workspace, workspace_id)
        if workspace is None or workspace.data_classification != "public":
            raise SystemExit("the combined benchmark needs a verified public-corpus workspace (decision D1)")
    clients: dict[str, Any] = {}
    for arm in arms:
        if arm.backend == "direct":
            clients[arm.name] = direct_client(settings)
        else:
            client: ModelMeshClient = modelmesh_client(settings.model_copy(update={"inference_mode": "modelmesh"}))
            if arm.policy is None:
                raise ValueError(f"{arm.name} has no routing policy")
            client.backend.policies = dict.fromkeys(
                (
                    "query_classification",
                    "query_rewriting",
                    "research_planning",
                    "evidence_assessment",
                    "evidence_synthesis",
                    "citation_verification",
                    "final_response",
                    "general",
                ),
                arm.policy,
            )
            client.negotiate()
            clients[arm.name] = client
    ops = modelmesh_ops(settings)

    def run_item(arm: Arm, item: Mapping[str, Any], run_id: str) -> ResearchState:
        with production_workflow(run_id, workspace_id, "hybrid_rrf") as deps:
            graph = build_graph(dataclasses.replace(deps, llm=clients[arm.name]))
            initial = ResearchState(
                run_id=run_id,
                workspace_id=workspace_id,
                question=str(item["question"]),
                data_policy="any",
                as_of=date.fromisoformat(str(item["as_of"])) if item.get("as_of") else None,
                budget=Budget(
                    llm_calls_left=settings.run_max_llm_calls,
                    tokens_left=settings.run_max_tokens,
                    deadline=time.time() + settings.run_deadline_s,
                ),
            )
            return run_to_end(graph, initial)

    def fetch(request_id: str) -> Mapping[str, Any] | None:
        return ops.request_record(request_id) if ops is not None else None

    return run_item, fetch
