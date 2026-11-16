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
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
