#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/suzuki_suite}"
SEED_START="${SEED_START:-0}"
TRIALS="${TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
REASONING_BATCH_SIZE="${REASONING_BATCH_SIZE:-3}"
GOLLUM_INIT_SIZE="${GOLLUM_INIT_SIZE:-10}"
REASONINGBO_LLM_MODEL="${REASONINGBO_LLM_MODEL:-kimi-k2.5}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export REASONINGBO_LLM_MODEL
export GOLLUM_DEVICE="${GOLLUM_DEVICE:-cpu}"

if [[ "${GOLLUM_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
fi

mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}"

echo "[suite] Running GOLLuM on Suzuki (${TRIALS} trials)"
conda run --no-capture-output -n gollum python   "${ROOT_DIR}/baseline/gollum/run_tabular_gollum.py"   --dataset suzuki   --trials "${TRIALS}"   --trial-start-index 1   --seed-start "${SEED_START}"   --total-budget "${TOTAL_BUDGET}"   --init-size "${GOLLUM_INIT_SIZE}"   --output-dir "${OUTPUT_ROOT}/gollum"

echo "[suite] Running Reasoning-BO on Suzuki (${TRIALS} trials)"
conda run --no-capture-output -n reasoning_bo python   "${ROOT_DIR}/baseline/Reasoning-BO/run_tabular_reasoning_bo.py"   --dataset suzuki   --trials 1   --trial-start-index 1   --seed-start "${SEED_START}"   --total-budget "${TOTAL_BUDGET}"   --reasoning-batch-size "${REASONING_BATCH_SIZE}"   --use-llm-initial-count   --allow-batch-overshoot   --output-dir "${OUTPUT_ROOT}/reasoning_bo"

echo "[suite] Completed. Outputs are under ${OUTPUT_ROOT}"
