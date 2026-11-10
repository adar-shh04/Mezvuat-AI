#!/usr/bin/env bash
# Start or resume relevance review. Usage: bash scripts/review.sh val   (then: bash scripts/review.sh test)
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/dev-env.sh
cd backend
PYTHONUTF8=1 uv run --no-sync mezvuat-dataset review "${1:-val}"
