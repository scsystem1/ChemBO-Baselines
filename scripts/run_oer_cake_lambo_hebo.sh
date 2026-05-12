#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-oer}"
OUT_ROOT="${OUT_ROOT:-outputs/oer_baselines}"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"

CAKE_CONDA_ENV="${CAKE_CONDA_ENV:-cake}"
LAMBO_CONDA_ENV="${LAMBO_CONDA_ENV:-lmabo}"
HEBO_CONDA_ENV="${HEBO_CONDA_ENV:-hebo}"

RUN_CAKE="${RUN_CAKE:-1}"
RUN_LAMBO="${RUN_LAMBO:-1}"
RUN_HEBO="${RUN_HEBO:-1}"

CAKE_ENABLE_LLM_EVOLUTION="${CAKE_ENABLE_LLM_EVOLUTION:-0}"
CAKE_MODEL="${CAKE_MODEL:-kimi-k2.5-thinking}"
LMABO_LLM_MODE="${LMABO_LLM_MODE:-none}"
LMABO_API_TYPE="${LMABO_API_TYPE:-gpt}"
LMABO_DEFAULT_ACQ="${LMABO_DEFAULT_ACQ:-UCB}"
LMABO_OPS_MODEL_NAME="${LMABO_OPS_MODEL_NAME:-Qwen/Qwen3-8B}"

print_config() {
  echo "[OER] dataset=${DATASET} target=objective goal=maximize"
  echo "[OER] out_root=${OUT_ROOT} trials=${TARGET_TRIALS} budget=${TOTAL_BUDGET} init_size=${INIT_SIZE}"
}

get_resume_info() {
  local method="$1"
  python -c "from pathlib import Path; import numpy as np, sys
method = sys.argv[1]
dataset = sys.argv[2]
out_root = Path(sys.argv[3])
total_budget = int(sys.argv[4])
results_path = out_root / method / dataset / f'{dataset}_{method}_results.npz'
completed = 0
has_incomplete = 0
if results_path.exists():
    payload = np.load(results_path)
    trace_lengths = payload['trace_lengths'].astype(int).tolist()
    completed = sum(1 for trace_len in trace_lengths if trace_len >= total_budget)
    has_incomplete = int(any(trace_len < total_budget for trace_len in trace_lengths))
print(f'{completed} {has_incomplete}')
" "$method" "${DATASET}" "${OUT_ROOT}" "${TOTAL_BUDGET}"
}

run_cake() {
  local resume_info existing_trials has_incomplete_trial remaining_trials trial_start_index seed_start
  resume_info="$(get_resume_info cake)"
  existing_trials="${resume_info%% *}"
  has_incomplete_trial="${resume_info##* }"
  remaining_trials=$((TARGET_TRIALS - existing_trials))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[CAKE][${DATASET^^}] Existing trials=${existing_trials}; nothing to run."
    return
  fi

  trial_start_index=$((existing_trials + 1))
  seed_start=$((existing_trials))
  echo "[CAKE][${DATASET^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local cake_args=(
    python baseline/cake/run_tabular_cake.py
    --dataset "${DATASET}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --output-dir "${OUT_ROOT}/cake/${DATASET}"
  )

  if [ "${CAKE_ENABLE_LLM_EVOLUTION}" = "1" ]; then
    cake_args+=(--enable-llm-evolution --model-name "${CAKE_MODEL}")
  fi
  if [ "${existing_trials}" -gt 0 ] || [ "${has_incomplete_trial}" = "1" ]; then
    cake_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete_trial}" = "1" ]; then
    cake_args+=(--replace-incomplete-last-trial)
  fi

  KIMI_MODEL="${CAKE_MODEL}" \
  conda run --no-capture-output -n "${CAKE_CONDA_ENV}" "${cake_args[@]}"
}

run_lambo() {
  local resume_info existing_trials has_incomplete_trial remaining_trials trial_start_index seed_start
  resume_info="$(get_resume_info lmabo)"
  existing_trials="${resume_info%% *}"
  has_incomplete_trial="${resume_info##* }"
  remaining_trials=$((TARGET_TRIALS - existing_trials))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[LAMBO][${DATASET^^}] Existing trials=${existing_trials}; nothing to run."
    return
  fi

  trial_start_index=$((existing_trials + 1))
  seed_start=$((existing_trials))
  echo "[LAMBO][${DATASET^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local lmabo_args=(
    python baseline/lmabo/run_tabular_lmabo.py
    --dataset "${DATASET}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --llm-mode "${LMABO_LLM_MODE}"
    --api-type "${LMABO_API_TYPE}"
    --default-acq "${LMABO_DEFAULT_ACQ}"
    --ops-model-name "${LMABO_OPS_MODEL_NAME}"
    --conversation-init-retries 3
    --conversation-init-delay-seconds 5
    --output-dir "${OUT_ROOT}/lmabo/${DATASET}"
  )

  if [ "${existing_trials}" -gt 0 ] || [ "${has_incomplete_trial}" = "1" ]; then
    lmabo_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete_trial}" = "1" ]; then
    lmabo_args+=(--replace-incomplete-last-trial)
  fi

  LMABO_LLM_MODEL="${LMABO_LLM_MODEL:-${CAKE_MODEL}}" \
  KIMI_MODEL="${KIMI_MODEL:-${CAKE_MODEL}}" \
  conda run --no-capture-output -n "${LAMBO_CONDA_ENV}" "${lmabo_args[@]}"
}

run_hebo() {
  local resume_info existing_trials has_incomplete_trial remaining_trials trial_start_index seed_start
  resume_info="$(get_resume_info hebo)"
  existing_trials="${resume_info%% *}"
  has_incomplete_trial="${resume_info##* }"
  remaining_trials=$((TARGET_TRIALS - existing_trials))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[HEBO][${DATASET^^}] Existing trials=${existing_trials}; nothing to run."
    return
  fi

  trial_start_index=$((existing_trials + 1))
  seed_start=$((existing_trials))
  echo "[HEBO][${DATASET^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local hebo_args=(
    python baseline/hebo/run_tabular_hebo.py
    --dataset "${DATASET}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --output-dir "${OUT_ROOT}/hebo/${DATASET}"
  )

  if [ "${existing_trials}" -gt 0 ] || [ "${has_incomplete_trial}" = "1" ]; then
    hebo_args+=(--append-to-existing)
  fi

  conda run --no-capture-output -n "${HEBO_CONDA_ENV}" "${hebo_args[@]}"
}

print_config

if [ "${RUN_CAKE}" = "1" ]; then
  run_cake
fi
if [ "${RUN_LAMBO}" = "1" ]; then
  run_lambo
fi
if [ "${RUN_HEBO}" = "1" ]; then
  run_hebo
fi
