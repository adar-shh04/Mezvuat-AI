# Source this from Git Bash before running uv on the dev machine: `. scripts/dev-env.sh`
# Keeps every cache off C: (see docs/ENVIRONMENT.md). Override MEZVUAT_CACHE_ROOT to relocate.
export MEZVUAT_CACHE_ROOT="${MEZVUAT_CACHE_ROOT:-X:/mezvuat-cache}"
export UV_CACHE_DIR="$MEZVUAT_CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$MEZVUAT_CACHE_ROOT/python"
export HF_HOME="$MEZVUAT_CACHE_ROOT/hf"
export MEZVUAT_DATA_ROOT="$MEZVUAT_CACHE_ROOT/data"
export MLFLOW_TRACKING_URI="sqlite:///$MEZVUAT_CACHE_ROOT/mlflow/mlflow.db"
export MLFLOW_ARTIFACTS_ROOT="$MEZVUAT_CACHE_ROOT/mlflow/artifacts"
export PATH="/c/Users/DELL/AppData/Roaming/Python/Python314/Scripts:$PATH"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$MEZVUAT_DATA_ROOT" "$MEZVUAT_CACHE_ROOT/mlflow/artifacts"
