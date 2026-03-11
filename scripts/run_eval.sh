#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from src.evaluation.benchmark import run_benchmark
print(run_benchmark())
PY
