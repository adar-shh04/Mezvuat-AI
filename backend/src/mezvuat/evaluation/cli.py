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
    args = parser.parse_args()
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
