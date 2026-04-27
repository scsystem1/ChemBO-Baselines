#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BOICL_LLM_MODEL="${BOICL_LLM_MODEL:-kimi-k2.5-thinking}"
export BOICL_BASE_URL="${BOICL_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export BOICL_EMBED_MODEL="${BOICL_EMBED_MODEL:-text-embedding-v4}"
export BOICL_EMBED_DIM="${BOICL_EMBED_DIM:-1536}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-100}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-100}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-100}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-100}"

echo "[BO-ICL][DAR] Starting run with model=${BOICL_LLM_MODEL}, embed_model=${BOICL_EMBED_MODEL}, budget=40, trials=3"
conda run --no-capture-output -n boicl python "${ROOT_DIR}/baseline/BO-ICL/run_tabular_experiment.py" \
  --dataset dar \
  --trials 3 \
  --total-budget 40 \
  --model "${BOICL_LLM_MODEL}" \
  --output-dir "${ROOT_DIR}/outputs/baseline_runs/boicl/dar"
