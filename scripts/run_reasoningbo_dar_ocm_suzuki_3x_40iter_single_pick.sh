#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/reasoning_bo_single_pick_suite}"
SEED_START="${SEED_START:-0}"
TRIALS="${TRIALS:-3}"
TRIAL_START_INDEX="${TRIAL_START_INDEX:-1}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
REASONING_BATCH_SIZE="${REASONING_BATCH_SIZE:-1}"

export REASONINGBO_LLM_MODEL="${REASONINGBO_LLM_MODEL:-kimi-k2.5-thinking}"
export REASONINGBO_BASE_URL="${REASONINGBO_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-40}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-40}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-40}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-40}"

mkdir -p "${OUTPUT_ROOT}"

run_dataset() {
  local dataset="$1"
  local -a extra_args=()

  if [[ "${dataset}" == "suzuki" ]]; then
    extra_args+=(--use-llm-initial-count --allow-batch-overshoot)
  fi

  echo "[ReasoningBO][${dataset^^}] Starting run with model=${REASONINGBO_LLM_MODEL}, total_budget=${TOTAL_BUDGET}, reasoning_batch_size=${REASONING_BATCH_SIZE}, trials=${TRIALS}"
  conda run --no-capture-output -n reasoning_bo python "${ROOT_DIR}/baseline/Reasoning-BO/run_tabular_reasoning_bo.py" \
    --dataset "${dataset}" \
    --trials "${TRIALS}" \
    --trial-start-index "${TRIAL_START_INDEX}" \
    --seed-start "${SEED_START}" \
    --total-budget "${TOTAL_BUDGET}" \
    --reasoning-batch-size "${REASONING_BATCH_SIZE}" \
    "${extra_args[@]}" \
    --output-dir "${OUTPUT_ROOT}/${dataset}"
}

run_dataset suzuki
run_dataset dar
run_dataset ocm

echo "[ReasoningBO][suite] Completed. Outputs are under ${OUTPUT_ROOT}"
