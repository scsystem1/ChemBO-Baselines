#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-outputs/hpobench_baselines}"
PROBLEMS="${PROBLEMS:-hpobench_rf_146606 hpobench_svm_146212 hpobench_xgb_146606 hpobench_nn_168912}"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-5}"
TARGET_COLUMN="${TARGET_COLUMN:-test_acc}"

CAKE_CONDA_ENV="${CAKE_CONDA_ENV:-cake}"
LMABO_CONDA_ENV="${LMABO_CONDA_ENV:-lmabo}"
HEBO_CONDA_ENV="${HEBO_CONDA_ENV:-hebo}"

RUN_CAKE="${RUN_CAKE:-1}"
RUN_LMABO="${RUN_LMABO:-1}"
RUN_HEBO="${RUN_HEBO:-1}"

DEEPSEEK_MODEL_NAME="${DEEPSEEK_MODEL_NAME:-DeepSeek-V4-Pro}"
LLM_API_KEY="${LLM_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}"
OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}}"

CAKE_ENABLE_LLM_EVOLUTION="${CAKE_ENABLE_LLM_EVOLUTION:-1}"
CAKE_MODEL="${CAKE_MODEL:-${DEEPSEEK_MODEL_NAME}}"
CAKE_API_BASE="${CAKE_API_BASE:-${OPENAI_COMPAT_BASE_URL}}"

LMABO_LLM_MODE="${LMABO_LLM_MODE:-api}"
LMABO_API_TYPE="${LMABO_API_TYPE:-gpt}"
LMABO_DEFAULT_ACQ="${LMABO_DEFAULT_ACQ:-UCB}"
LMABO_OPS_MODEL_NAME="${LMABO_OPS_MODEL_NAME:-Qwen/Qwen3-8B}"
LMABO_LLM_MODEL="${LMABO_LLM_MODEL:-${DEEPSEEK_MODEL_NAME}}"

print_config() {
  echo "[HPOBench] problems=${PROBLEMS}"
  echo "[HPOBench] out_root=${OUT_ROOT} trials=${TARGET_TRIALS} budget=${TOTAL_BUDGET} init_size=${INIT_SIZE}"
  echo "[HPOBench] run_cake=${RUN_CAKE} run_lmabo=${RUN_LMABO} run_hebo=${RUN_HEBO}"
  echo "[HPOBench] llm_model=${DEEPSEEK_MODEL_NAME} api_base=${OPENAI_COMPAT_BASE_URL}"
}

problem_data_path() {
  local problem="$1"
  printf '%s/data/HPOBench/%s.csv' "${ROOT_DIR}" "${problem}"
}

ensure_problem_exists() {
  local problem="$1"
  local data_path
  data_path="$(problem_data_path "${problem}")"
  if [ ! -f "${data_path}" ]; then
    echo "[HPOBench][${problem}] Missing dataset file: ${data_path}" >&2
    return 1
  fi
}

get_resume_info() {
  local method="$1"
  local dataset_name="$2"
  python -c "from pathlib import Path; import numpy as np, sys
method = sys.argv[1]
dataset = sys.argv[2]
out_root = Path(sys.argv[3])
total_budget = int(sys.argv[4])
results_path = out_root / method / dataset / f'{dataset}_{method}_results.npz'
completed = 0
recorded = 0
has_incomplete = 0
if results_path.exists():
    payload = np.load(results_path)
    trace_lengths = payload['trace_lengths'].astype(int).tolist()
    recorded = len(trace_lengths)
    completed = sum(1 for trace_len in trace_lengths if trace_len >= total_budget)
    has_incomplete = int(any(trace_len < total_budget for trace_len in trace_lengths))
print(f'{completed} {recorded} {has_incomplete}')
" "${method}" "${dataset_name}" "${OUT_ROOT}" "${TOTAL_BUDGET}"
}

run_cake_problem() {
  local problem="$1"
  local dataset_name="$problem"
  local data_path
  data_path="$(problem_data_path "${problem}")"

  local resume_info existing_completed existing_recorded has_incomplete remaining_trials
  resume_info="$(get_resume_info cake "${dataset_name}")"
  read -r existing_completed existing_recorded has_incomplete <<<"${resume_info}"
  remaining_trials=$((TARGET_TRIALS - existing_completed))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[CAKE][${dataset_name}] Existing completed trials=${existing_completed}; nothing to run."
    return
  fi

  local trial_start_index seed_start
  trial_start_index=$((existing_completed + 1))
  seed_start="${existing_completed}"
  echo "[CAKE][${dataset_name}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local cake_args=(
    python baseline/cake/run_tabular_cake.py
    --dataset "${dataset_name}"
    --data-path "${data_path}"
    --target-column "${TARGET_COLUMN}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --output-dir "${OUT_ROOT}/cake/${dataset_name}"
  )

  if [ "${CAKE_ENABLE_LLM_EVOLUTION}" = "1" ]; then
    cake_args+=(--enable-llm-evolution --model-name "${CAKE_MODEL}")
    if [ -n "${CAKE_API_BASE}" ]; then
      cake_args+=(--api-base "${CAKE_API_BASE}")
    fi
  fi
  if [ "${existing_recorded}" -gt 0 ] || [ "${has_incomplete}" = "1" ]; then
    cake_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete}" = "1" ]; then
    cake_args+=(--replace-incomplete-last-trial)
  fi

  OPENAI_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_BASE="${OPENAI_COMPAT_BASE_URL}" \
  DEEPSEEK_MODEL_NAME="${CAKE_MODEL}" \
  KIMI_MODEL="${CAKE_MODEL}" \
  KIMI_BASE_URL="${OPENAI_COMPAT_BASE_URL}" \
  CAKE_BASE_URL="${CAKE_API_BASE}" \
  conda run --no-capture-output -n "${CAKE_CONDA_ENV}" "${cake_args[@]}"
}

run_lmabo_problem() {
  local problem="$1"
  local dataset_name="$problem"
  local data_path
  data_path="$(problem_data_path "${problem}")"

  local resume_info existing_completed existing_recorded has_incomplete remaining_trials
  resume_info="$(get_resume_info lmabo "${dataset_name}")"
  read -r existing_completed existing_recorded has_incomplete <<<"${resume_info}"
  remaining_trials=$((TARGET_TRIALS - existing_completed))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[LMABO][${dataset_name}] Existing completed trials=${existing_completed}; nothing to run."
    return
  fi

  local trial_start_index seed_start
  trial_start_index=$((existing_completed + 1))
  seed_start="${existing_completed}"
  echo "[LMABO][${dataset_name}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local lmabo_args=(
    python baseline/lmabo/run_tabular_lmabo.py
    --dataset "${dataset_name}"
    --data-path "${data_path}"
    --target-column "${TARGET_COLUMN}"
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
    --output-dir "${OUT_ROOT}/lmabo/${dataset_name}"
  )

  if [ "${existing_recorded}" -gt 0 ] || [ "${has_incomplete}" = "1" ]; then
    lmabo_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete}" = "1" ]; then
    lmabo_args+=(--replace-incomplete-last-trial)
  fi

  OPENAI_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_BASE="${OPENAI_COMPAT_BASE_URL}" \
  DEEPSEEK_MODEL_NAME="${LMABO_LLM_MODEL}" \
  LMABO_LLM_MODEL="${LMABO_LLM_MODEL}" \
  KIMI_MODEL="${LMABO_LLM_MODEL}" \
  KIMI_BASE_URL="${OPENAI_COMPAT_BASE_URL}" \
  conda run --no-capture-output -n "${LMABO_CONDA_ENV}" "${lmabo_args[@]}"
}

run_hebo_problem() {
  local problem="$1"
  local dataset_name="$problem"
  local data_path
  data_path="$(problem_data_path "${problem}")"

  local resume_info existing_completed existing_recorded has_incomplete remaining_trials
  resume_info="$(get_resume_info hebo "${dataset_name}")"
  read -r existing_completed existing_recorded has_incomplete <<<"${resume_info}"

  if [ "${has_incomplete}" = "1" ]; then
    echo "[HEBO][${dataset_name}] Found an incomplete recorded trial in ${OUT_ROOT}/hebo/${dataset_name}." >&2
    echo "[HEBO][${dataset_name}] HEBO runner cannot replace incomplete trials automatically; please remove or fix that result first." >&2
    return 1
  fi

  remaining_trials=$((TARGET_TRIALS - existing_completed))
  if [ "${remaining_trials}" -le 0 ]; then
    echo "[HEBO][${dataset_name}] Existing completed trials=${existing_completed}; nothing to run."
    return
  fi

  local trial_start_index seed_start
  trial_start_index=$((existing_completed + 1))
  seed_start="${existing_completed}"
  echo "[HEBO][${dataset_name}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local hebo_args=(
    python baseline/hebo/run_tabular_hebo.py
    --dataset "${dataset_name}"
    --data-path "${data_path}"
    --target-column "${TARGET_COLUMN}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --output-dir "${OUT_ROOT}/hebo/${dataset_name}"
  )

  if [ "${existing_recorded}" -gt 0 ]; then
    hebo_args+=(--append-to-existing)
  fi

  conda run --no-capture-output -n "${HEBO_CONDA_ENV}" "${hebo_args[@]}"
}

run_problem() {
  local problem="$1"
  ensure_problem_exists "${problem}"
  echo "[HPOBench][${problem}] Starting"

  if [ "${RUN_CAKE}" = "1" ]; then
    run_cake_problem "${problem}"
  fi
  if [ "${RUN_LMABO}" = "1" ]; then
    run_lmabo_problem "${problem}"
  fi
  if [ "${RUN_HEBO}" = "1" ]; then
    run_hebo_problem "${problem}"
  fi
}

print_config

for problem in ${PROBLEMS}; do
  run_problem "${problem}"
done
