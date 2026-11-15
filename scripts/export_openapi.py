"""Write the backend's OpenAPI document, which the frontend's API types are generated from (M7).

Run from backend/: `uv run --no-sync python ../scripts/export_openapi.py [--out ../frontend/openapi.json]`.
`--check` exits 1 when the file on disk differs, so CI catches a backend change without regenerated types.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "frontend" / "openapi.json"


def render() -> str:
    from mezvuat.api.app import create_app

    return json.dumps(create_app().openapi(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date instead of writing")
    args = parser.parse_args()
    document = render()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != document:
            print(f"{args.out} is out of date: run scripts/export_openapi.py and `npm run gen:api`", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0
    args.out.write_text(document, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
