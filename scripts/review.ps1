# Start or resume review from PowerShell. Usage: .\scripts\review.ps1 val   (then: .\scripts\review.ps1 test)
# Temporal question set (decision D11): .\scripts\review.ps1 temporal  (writes data/domain/temporal/v1/review.human.jsonl)
# NLI calibration pairs (M5): .\scripts\review.ps1 nli  (writes data/domain/nli/v1/review.human.jsonl; no model suggestion is shown)
param([ValidateSet("val", "test", "temporal", "nli")][string]$Split = "val")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$cache = if ($env:MEZVUAT_CACHE_ROOT) { $env:MEZVUAT_CACHE_ROOT } else { "X:/mezvuat-cache" }
$env:UV_CACHE_DIR = "$cache/uv"
$env:UV_PYTHON_INSTALL_DIR = "$cache/python"
$env:HF_HOME = "$cache/hf"
$env:MEZVUAT_DATA_ROOT = "$cache/data"
$env:PYTHONUTF8 = "1"
$uv = "$env:APPDATA/Python/Python314/Scripts/uv.exe"
if (-not (Test-Path $uv)) { $uv = "uv" }
Push-Location "$repo/backend"
try { & $uv run --no-sync mezvuat-dataset review $Split } finally { Pop-Location }
