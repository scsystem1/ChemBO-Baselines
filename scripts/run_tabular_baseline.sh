#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_CONDA_ENV="${RUNNER_CONDA_ENV:-CAS}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --baseline <name> --dataset <name> [runner args] [baseline args]" >&2
  echo "Example: $0 --baseline gollum --dataset dar --trials 3 --total-budget 40 --init-size 10" >&2
  exit 1
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

conda run --no-capture-output -n "${RUNNER_CONDA_ENV}" python "${ROOT_DIR}/run_tabular_baseline.py" "$@"
