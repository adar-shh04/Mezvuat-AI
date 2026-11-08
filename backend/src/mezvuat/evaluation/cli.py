"""Experiment CLI: `mezvuat-exp run <spec.yaml>`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mezvuat.core.logging import configure_logging
from mezvuat.evaluation.runner import load_spec, run_experiment


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="mezvuat-exp")
    sub = parser.add_subparsers(required=True, dest="command")
    run = sub.add_parser("run")
    run.add_argument("spec", type=Path)
    run.add_argument("--out", type=Path, default=None, help="write a JSON summary here")
    args = parser.parse_args()
    if args.command == "run":
        root = Path(os.environ.get("MLFLOW_ARTIFACTS_ROOT", "mlartifacts")) / "local-runs"
        summaries = run_experiment(load_spec(args.spec), artifact_root=root)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summaries, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
