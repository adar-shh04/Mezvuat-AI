"""Answer-level evaluation over the four retrieval configurations (M6, ADR-0012). NOT RUN YET: the owner's labels are
not finished, and every number this produces must come from a real run.

One command (`mezvuat-exp answer-eval`) runs every gold item through the full research workflow once per retrieval
configuration, scores each run with Mezvuat's metrics (`answer_metrics`) and, unless `--no-ragas`, with RAGAS behind
the pinned judge (`ragas_metrics`), and writes:
- an MLflow parent run with one nested run per configuration (params, aggregate metrics, the per-item rows and the
  Langfuse trace ids as artifacts, so MLflow links to Langfuse without copying traces);
- `results.json` and `report.md`, generated from the same rows;
- optionally (`--post-feedback`), per-request labels for ModelMesh through the outbox poster: RAGAS faithfulness on
  the synthesis request and answer relevancy on the final answer (`attribution: direct`), `task_success` on every
  request of the run (`attribution: workflow`), all with `metric_version`, `eval_run_id = "mlflow:<run id>"`,
  `dataset_version` and `labelled_at`.

Decision D1: the workspace must be a verified public-corpus workspace (judge calls carry evidence).
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from mezvuat.core.ids import new_id
from mezvuat.evaluation.answer_metrics import METRIC_VERSION, abstained, answer_metrics
from mezvuat.evaluation.domain import read_jsonl
from mezvuat.inference.modelmesh import FeedbackItem
from mezvuat.orchestration.state import ResearchState

log = logging.getLogger(__name__)

CONFIGS = ("bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank")
REPORT_VERSION = "answer-eval-v1"
# Which request a per-request RAGAS label belongs to (ADR-0011 feedback table).
_DIRECT_LABELS = {"faithfulness": "evidence_synthesis", "answer_relevancy": "final_response"}


class Scorer(Protocol):
    def score(
        self, question: str, answer: str | None, contexts: Sequence[str], reference: str | None
    ) -> dict[str, float]: ...


class Tracker(Protocol):
    """Experiment tracking. `MlflowTracker` in production; tests use a fake."""

    def run(self, name: str, *, nested: bool = False) -> Any: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...

    def log_json(self, data: Any, name: str) -> None: ...

    def run_id(self) -> str: ...


class MlflowTracker:
    """MLflow with the URI from `MLFLOW_TRACKING_URI` (dev: SQLite under X:/mezvuat-cache, see scripts/dev-env.sh)."""

    def __init__(self, experiment: str = "answer-eval") -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[None]:
        with self._mlflow.start_run(run_name=name, nested=nested):
            yield

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics(dict(metrics))

    def log_json(self, data: Any, name: str) -> None:
        self._mlflow.log_dict(data, name)

    def run_id(self) -> str:
        run = self._mlflow.active_run()
        return str(run.info.run_id) if run else ""


RunItem = Callable[[str, Mapping[str, Any], str], ResearchState]


def dataset_version(items: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda i: str(i["item_id"])):
        digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode())
    return f"gold-{digest.hexdigest()[:12]}"


def load_temporal_gold(temporal_dir: Path) -> list[dict[str, Any]]:
    """Reviewed temporal items (D11): the drafts merged with the owner's latest decision. Read-only; rejected and
    unreviewed items are left out. The answer the owner accepted or wrote becomes the reference answer."""
    drafts = {row["item_id"]: row for row in read_jsonl(temporal_dir / "drafts.jsonl")}
    review_path = temporal_dir / "review.human.jsonl"
    decided: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(review_path) if review_path.exists() else []:
        if record.get("retracted"):
            decided.pop(record["item_id"], None)
        else:
            decided[record["item_id"]] = record
    gold = []
    for item_id, record in sorted(decided.items()):
        if record.get("decision") not in ("accept", "edit", "write") or item_id not in drafts:
            continue
        draft = drafts[item_id]
        gold.append(
            {
                "item_id": item_id,
                "question": draft["question"],
                "reference_answer": record.get("answer"),
                "expected_behavior": record.get("expected_behavior", draft.get("expected_behavior")),
                "expected_versions": draft.get("expected_versions"),
                "as_of": draft.get("as_of"),
                "status": "reviewed",
                "source": "temporal-v1",
            }
        )
    return gold


def validate_items(items: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(i.get("item_id")) for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item_id in the gold set")
    for item in items:
        if not item.get("question"):
            raise ValueError(f"{item.get('item_id')}: no question")
        if item.get("status") in ("draft", "rejected"):
            raise ValueError(f"{item['item_id']}: unreviewed or rejected items are not gold (D11)")


def contexts_shown(state: ResearchState) -> list[str]:
    return [state.evidence[e].text for e in state.shown_evidence_ids if e in state.evidence]


@dataclass
class EvalResult:
    eval_run_id: str
    dataset_version: str
    judge_model: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        per_config: dict[str, Any] = {}
        for config in dict.fromkeys(r["config"] for r in self.rows):
            rows = [r for r in self.rows if r["config"] == config]
            names = sorted({k for r in rows for k in r["metrics"]})
            metrics = {}
            for name in names:
                values = [r["metrics"][name] for r in rows if name in r["metrics"]]
                metrics[name] = {"mean": statistics.fmean(values), "n": len(values)}
            walls = sorted(r["wall_ms"] for r in rows)
            per_config[config] = {
                "items": len(rows),
                "metrics": metrics,
                "wall_ms_p50": statistics.median(walls),
                "wall_ms_p95": walls[min(len(walls) - 1, round(0.95 * (len(walls) - 1)))],
            }
        return {
            "report_version": REPORT_VERSION,
            "eval_run_id": self.eval_run_id,
            "dataset_version": self.dataset_version,
            "judge_model": self.judge_model,
            "per_config": per_config,
        }


def evaluate(
    items: Sequence[Mapping[str, Any]],
    configs: Sequence[str],
    run_item: RunItem,
    tracker: Tracker,
    *,
    scorer: Scorer | None = None,
    judge_model: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvalResult:
    validate_items(items)
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        raise ValueError(f"unknown retrieval configurations: {sorted(unknown)}")
    version = dataset_version(items)
    with tracker.run("answer-eval"):
        result = EvalResult(eval_run_id=f"mlflow:{tracker.run_id()}", dataset_version=version, judge_model=judge_model)
        tracker.log_params(
            {
                "configs": ",".join(configs),
                "items": len(items),
                "dataset_version": version,
                "judge_model": judge_model or "none",
                "metric_version": METRIC_VERSION,
                "report_version": REPORT_VERSION,
            }
        )
        for config in configs:
            with tracker.run(config, nested=True):
                rows = []
                for item in items:
                    # A fresh id per run: the Langfuse trace id is derived from it, so re-evaluations never merge.
                    run_id = new_id("run")
                    started = clock()
                    state = run_item(config, item, run_id)
                    wall_ms = int((clock() - started) * 1000)
                    metrics = answer_metrics(state, item)
                    if scorer is not None and state.status == "completed":
                        answer = None if abstained(state) or state.answer is None else state.answer.markdown
                        metrics.update(
                            scorer.score(
                                str(item["question"]), answer, contexts_shown(state), item.get("reference_answer")
                            )
                        )
                    rows.append(
                        {
                            "item_id": item["item_id"],
                            "config": config,
                            "run_id": run_id,
                            "status": state.status,
                            "error": state.error.code if state.error else None,
                            "metrics": metrics,
                            "wall_ms": wall_ms,
                            "langfuse_trace_ids": sorted(
                                {c.langfuse_trace_id for c in state.llm_calls if c.langfuse_trace_id}
                            ),
                            "calls": [
                                {
                                    "task_type": c.task_type,
                                    "status": c.status,
                                    "modelmesh_request_id": c.modelmesh_request_id,
                                }
                                for c in state.llm_calls
                            ],
                        }
                    )
                result.rows.extend(rows)
                summary = result.summary()["per_config"][config]
                tracker.log_metrics({f"{k}_mean": v["mean"] for k, v in summary["metrics"].items()})
                tracker.log_metrics({"wall_ms_p50": summary["wall_ms_p50"], "wall_ms_p95": summary["wall_ms_p95"]})
                tracker.log_json(rows, f"rows_{config}.json")
                tracker.log_json(
                    sorted({t for r in rows for t in r["langfuse_trace_ids"]}), f"langfuse_traces_{config}.json"
                )
        tracker.log_json(result.summary(), "summary.json")
    return result


def feedback_items(
    result: EvalResult, *, evaluator: str, ragas_metric_version: str, labelled_at: datetime
) -> list[FeedbackItem]:
    """ModelMesh labels for the evaluated requests (contract §5, ADR-0011). Only ModelMesh requests get labels."""
    items: list[FeedbackItem] = []
    for row in result.rows:
        ok_calls = [c for c in row["calls"] if c["status"] == "ok" and c["modelmesh_request_id"]]
        common = {
            "label_source": "automated_eval",
            "evaluator": evaluator,
            "eval_run_id": result.eval_run_id[:128],
            "dataset_version": result.dataset_version,
            "labelled_at": labelled_at,
        }
        for metric, task_type in _DIRECT_LABELS.items():
            score = row["metrics"].get(metric)
            target = next((c for c in reversed(ok_calls) if c["task_type"] == task_type), None)
            if score is None or target is None:
                continue
            items.append(
                FeedbackItem(
                    request_id=target["modelmesh_request_id"],
                    metric=metric,
                    metric_version=ragas_metric_version,
                    score=min(max(float(score), 0.0), 1.0),
                    attribution="direct",
                    task_type=task_type,
                    **common,
                )
            )
        metrics = row["metrics"]
        if "abstention_correct" in metrics or "citation_validity" in metrics:
            success = (
                metrics.get("completed") == 1.0
                and metrics.get("abstention_correct", 1.0) == 1.0
                and metrics.get("citation_validity", 1.0) == 1.0
            )
            for call in ok_calls:
                items.append(
                    FeedbackItem(
                        request_id=call["modelmesh_request_id"],
                        metric="task_success",
                        metric_version=METRIC_VERSION,
                        score=1.0 if success else 0.0,
                        accepted=success,
                        attribution="workflow",
                        task_type=call["task_type"],
                        **common,
                    )
                )
    return items


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Answer-level evaluation (M6)",
        "",
        f"Generated by `{REPORT_VERSION}` from `results.json`. MLflow run `{summary['eval_run_id']}`, dataset "
        f"`{summary['dataset_version']}`, judge `{summary['judge_model'] or 'none (no RAGAS)'}`.",
        "",
    ]
    names = sorted({m for c in summary["per_config"].values() for m in c["metrics"]})
    header = ["config", "items", *names, "p50 wall (ms)", "p95 wall (ms)"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for config, data in summary["per_config"].items():
        cells = [config, str(data["items"])]
        for name in names:
            metric = data["metrics"].get(name)
            cells.append(f"{metric['mean']:.3f} (n={metric['n']})" if metric else "n/a")
        cells += [f"{data['wall_ms_p50']:.0f}", f"{data['wall_ms_p95']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, result: EvalResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "rows": result.rows}, indent=1), encoding="utf-8", newline="\n"
    )
    report = out_dir / "report.md"
    report.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    return report


def production_runner(workspace_id: str) -> RunItem:
    """The real stack: the API's workflow (its LLM client, retrieval and tracer) for each configuration."""
    from mezvuat.api.deps import get_tracer, production_workflow
    from mezvuat.core.config import get_settings
    from mezvuat.orchestration import Budget, build_graph, run_to_end
    from mezvuat.persistence.db import get_session_factory
    from mezvuat.persistence.models import Workspace

    settings = get_settings()
    with get_session_factory()() as session:
        workspace = session.get(Workspace, workspace_id)
        if workspace is None or workspace.data_classification != "public":
            raise SystemExit("answer evaluation needs a verified public-corpus workspace (decision D1)")
    tracer = get_tracer()

    def run_item(config: str, item: Mapping[str, Any], run_id: str) -> ResearchState:
        with production_workflow(run_id, workspace_id, config) as deps:  # type: ignore[arg-type]
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
            with tracer.run(run_id, metadata={"workspace_id": workspace_id, "retrieval_mode": config}) as callbacks:
                return run_to_end(build_graph(deps), initial, callbacks=callbacks)

    return run_item


def now() -> datetime:
    return datetime.now(UTC)
