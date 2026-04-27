#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HOME="${HF_HOME:-/data/shared/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export GOLLUM_CUDA_VISIBLE_DEVICES="${GOLLUM_CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GOLLUM_CUDA_VISIBLE_DEVICES}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-100}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-100}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-100}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-100}"

echo "[GOLLuM][DAR] Starting run on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, budget=40, trials=3"
conda run --no-capture-output -n gollum python "${ROOT_DIR}/baseline/gollum/run_tabular_gollum.py" \
  --dataset dar \
  --trials 3 \
  --total-budget 40 \
  --output-dir "${ROOT_DIR}/outputs/baseline_runs/gollum/dar"
