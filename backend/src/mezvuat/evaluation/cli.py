"""Experiment CLI: `mezvuat-exp run <spec.yaml>`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mezvuat.core.logging import configure_logging
from mezvuat.evaluation.report import render
from mezvuat.evaluation.runner import load_spec, run_experiment


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="mezvuat-exp")
    sub = parser.add_subparsers(required=True, dest="command")
    run = sub.add_parser("run")
    run.add_argument("spec", type=Path)
    run.add_argument("--out", type=Path, default=None, help="write a JSON summary here")
    report = sub.add_parser("report")
    report.add_argument("summaries", type=Path, nargs="+")
    report.add_argument("--out", type=Path, required=True)
    bench = sub.add_parser("combined-benchmark", help="direct vs ModelMesh routing policies (M8); see the module doc")
    bench.add_argument("--items", type=Path, required=True, help="gold eval items, JSONL with item_id and question")
    bench.add_argument("--workspace", required=True, help="a verified public-corpus workspace id (decision D1)")
    bench.add_argument(
        "--arms",
        default="direct,modelmesh:cost_optimized,modelmesh:quality_balanced,modelmesh:latency_optimized",
        help="comma-separated: direct and modelmesh:<policy>",
    )
    bench.add_argument("--limit", type=int, default=None, help="only the first N items")
    bench.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    bench.add_argument("--out", type=Path, required=True, help="directory for results.json and report.md")
    gold = sub.add_parser("export-gold", help="reviewed temporal items as gold JSONL (read-only on the review file)")
    gold.add_argument("--temporal-dir", type=Path, default=Path("../data/domain/temporal/v1"))
    gold.add_argument("--out", type=Path, required=True)
    answers = sub.add_parser("answer-eval", help="answer-level evaluation over the retrieval configurations (M6)")
    answers.add_argument("--items", type=Path, required=True, help="gold items JSONL (see evaluation/answer_metrics)")
    answers.add_argument("--workspace", required=True, help="a verified public-corpus workspace id (decision D1)")
    answers.add_argument("--configs", default="bm25,dense,hybrid_rrf,hybrid_rrf_rerank")
    answers.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL"), help="pinned judge model id")
    answers.add_argument(
        "--judge-backend", choices=["direct", "modelmesh"], default=os.environ.get("EVAL_JUDGE_BACKEND") or "direct"
    )
    answers.add_argument("--no-ragas", action="store_true", help="Mezvuat metrics only, no judge calls")
    answers.add_argument("--post-feedback", action="store_true", help="queue labels for ModelMesh's ModelMesh requests")
    answers.add_argument("--limit", type=int, default=None)
    answers.add_argument("--out", type=Path, required=True, help="directory for results.json and report.md")
    args = parser.parse_args()
    if args.command == "export-gold":
        from mezvuat.evaluation.answer_eval import load_temporal_gold
        from mezvuat.evaluation.domain import write_jsonl

        rows = load_temporal_gold(args.temporal_dir)
        write_jsonl(args.out, rows)
        print(f"{len(rows)} reviewed items -> {args.out}")
        return
    if args.command == "answer-eval":
        _answer_eval(args)
        return
    if args.command == "combined-benchmark":
        from mezvuat.evaluation import combined_benchmark as cb

        arms = [cb.Arm.parse(a.strip()) for a in args.arms.split(",") if a.strip()]
        items = cb.read_items(args.items)[: args.limit]
        run_item, fetch = cb.production_runner(args.workspace, arms)
        result = cb.run_benchmark(items, arms, run_item, fetch)
        mlflow_runs = None if args.no_mlflow else cb.log_to_mlflow(result.summary())
        print(cb.write_outputs(args.out, result, mlflow_runs))
        return
    if args.command == "report":
        args.out.write_text(render(args.summaries), encoding="utf-8", newline="\n")
        return
    if args.command == "run":
        root = Path(os.environ.get("MLFLOW_ARTIFACTS_ROOT", "mlartifacts")) / "local-runs"
        summaries = run_experiment(load_spec(args.spec), artifact_root=root)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summaries, indent=1), encoding="utf-8")


def _answer_eval(args: argparse.Namespace) -> None:
    from mezvuat.core.config import get_settings
    from mezvuat.evaluation import answer_eval as ae
    from mezvuat.evaluation.domain import read_jsonl

    items = read_jsonl(args.items)[: args.limit]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    tracker = ae.MlflowTracker()
    scorer = None
    if not args.no_ragas:
        if not args.judge_model:
            raise SystemExit("RAGAS needs a pinned judge: --judge-model (or EVAL_JUDGE_MODEL), or pass --no-ragas")
        from mezvuat.api.deps import LazyEmbedder
        from mezvuat.evaluation.judge import judge_from_settings
        from mezvuat.evaluation.ragas_metrics import RagasScorer

        judge = judge_from_settings(get_settings(), args.judge_model, "answer-eval", args.judge_backend)
        scorer = RagasScorer(judge, LazyEmbedder().embed_query)
    result = ae.evaluate(
        items, configs, ae.production_runner(args.workspace), tracker, scorer=scorer, judge_model=args.judge_model
    )
    print(ae.write_outputs(args.out, result))
    if args.post_feedback:
        from mezvuat.evaluation.ragas_metrics import ragas_version
        from mezvuat.persistence.db import get_session_factory
        from mezvuat.persistence.modelmesh_outbox import enqueue_feedback

        version = ragas_version()
        labels = ae.feedback_items(
            result,
            evaluator=f"mezvuat-eval/ragas-{version}+judge:{args.judge_model}"[:128],
            ragas_metric_version=f"ragas-{version}",
            labelled_at=ae.now(),
        )
        print(f"{enqueue_feedback(get_session_factory(), labels)} labels queued for ModelMesh")


if __name__ == "__main__":
    main()
