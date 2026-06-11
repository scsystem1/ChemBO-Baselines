#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-outputs/formal_runs_ds}"
DATASETS="${DATASETS:-dar ocm oer suzuki}"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"

CAKE_CONDA_ENV="${CAKE_CONDA_ENV:-cake}"
LMABO_CONDA_ENV="${LMABO_CONDA_ENV:-lmabo}"

RUN_CAKE="${RUN_CAKE:-1}"
RUN_LMABO="${RUN_LMABO:-1}"

DEEPSEEK_MODEL_NAME="${DEEPSEEK_MODEL_NAME:-DeepSeek-V4-Pro}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
LLM_API_KEY="${LLM_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}"

CAKE_OUTPUT_SUBDIR="${CAKE_OUTPUT_SUBDIR:-cake}"
CAKE_ENABLE_LLM_EVOLUTION="${CAKE_ENABLE_LLM_EVOLUTION:-1}"
CAKE_MODEL="${CAKE_MODEL:-${DEEPSEEK_MODEL_NAME}}"
CAKE_API_BASE="${CAKE_API_BASE:-${DEEPSEEK_API_BASE}}"
CAKE_SEED_OFFSET="${CAKE_SEED_OFFSET:-42}"

LMABO_OUTPUT_SUBDIR="${LMABO_OUTPUT_SUBDIR:-lmabo}"
LMABO_LLM_MODE="${LMABO_LLM_MODE:-api}"
LMABO_API_TYPE="${LMABO_API_TYPE:-gpt}"
LMABO_DEFAULT_ACQ="${LMABO_DEFAULT_ACQ:-UCB}"
LMABO_OPS_MODEL_NAME="${LMABO_OPS_MODEL_NAME:-Qwen/Qwen3-8B}"
LMABO_LLM_MODEL="${LMABO_LLM_MODEL:-${DEEPSEEK_MODEL_NAME}}"
LMABO_SEED_OFFSET="${LMABO_SEED_OFFSET:-0}"

print_config() {
  echo "[Formal-DS] datasets=${DATASETS}"
  echo "[Formal-DS] out_root=${OUT_ROOT} trials=${TARGET_TRIALS} budget=${TOTAL_BUDGET} init_size=${INIT_SIZE}"
  echo "[Formal-DS] run_cake=${RUN_CAKE} run_lmabo=${RUN_LMABO}"
  echo "[Formal-DS] model=${DEEPSEEK_MODEL_NAME} api_base=${DEEPSEEK_API_BASE}"
  echo "[Formal-DS] cake_dir=${OUT_ROOT}/${CAKE_OUTPUT_SUBDIR}/<dataset>"
  echo "[Formal-DS] lmabo_dir=${OUT_ROOT}/${LMABO_OUTPUT_SUBDIR}/<dataset>"
}

get_resume_info() {
  local output_subdir="$1"
  local method="$2"
  local dataset="$3"
  python -c "from pathlib import Path; import numpy as np, sys
out_root = Path(sys.argv[1])
output_subdir = sys.argv[2]
method = sys.argv[3]
dataset = sys.argv[4]
total_budget = int(sys.argv[5])
results_path = out_root / output_subdir / dataset / f'{dataset}_{method}_results.npz'
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
" "${OUT_ROOT}" "${output_subdir}" "${method}" "${dataset}" "${TOTAL_BUDGET}"
}

run_cake_dataset() {
  local ds="$1"
  local resume_info existing_completed existing_recorded has_incomplete remaining_trials
  resume_info="$(get_resume_info "${CAKE_OUTPUT_SUBDIR}" cake "${ds}")"
  read -r existing_completed existing_recorded has_incomplete <<<"${resume_info}"
  remaining_trials=$((TARGET_TRIALS - existing_completed))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[CAKE][${ds^^}] Existing completed trials=${existing_completed}; nothing to run."
    return
  fi

  local trial_start_index seed_start
  trial_start_index=$((existing_completed + 1))
  seed_start=$((existing_completed + CAKE_SEED_OFFSET))
  echo "[CAKE][${ds^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local cake_args=(
    python baseline/cake/run_tabular_cake.py
    --dataset "${ds}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --output-dir "${OUT_ROOT}/${CAKE_OUTPUT_SUBDIR}/${ds}"
  )

  if [ "${CAKE_ENABLE_LLM_EVOLUTION}" = "1" ]; then
    cake_args+=(--enable-llm-evolution --model-name "${CAKE_MODEL}" --api-base "${CAKE_API_BASE}")
  fi
  if [ "${existing_recorded}" -gt 0 ] || [ "${has_incomplete}" = "1" ]; then
    cake_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete}" = "1" ]; then
    cake_args+=(--replace-incomplete-last-trial)
  fi

  OPENAI_API_KEY="${LLM_API_KEY}" \
  DASHSCOPE_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_KEY="${LLM_API_KEY}" \
  KIMI_MODEL="${CAKE_MODEL}" \
  KIMI_BASE_URL="${DEEPSEEK_API_BASE}" \
  CAKE_BASE_URL="${CAKE_API_BASE}" \
  conda run --no-capture-output -n "${CAKE_CONDA_ENV}" "${cake_args[@]}"
}

run_lmabo_dataset() {
  local ds="$1"
  local resume_info existing_completed existing_recorded has_incomplete remaining_trials
  resume_info="$(get_resume_info "${LMABO_OUTPUT_SUBDIR}" lmabo "${ds}")"
  read -r existing_completed existing_recorded has_incomplete <<<"${resume_info}"
  remaining_trials=$((TARGET_TRIALS - existing_completed))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[LMABO][${ds^^}] Existing completed trials=${existing_completed}; nothing to run."
    return
  fi

  local trial_start_index seed_start
  trial_start_index=$((existing_completed + 1))
  seed_start=$((existing_completed + LMABO_SEED_OFFSET))
  echo "[LMABO][${ds^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local lmabo_args=(
    python baseline/lmabo/run_tabular_lmabo.py
    --dataset "${ds}"
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
    --output-dir "${OUT_ROOT}/${LMABO_OUTPUT_SUBDIR}/${ds}"
  )

  if [ "${existing_recorded}" -gt 0 ] || [ "${has_incomplete}" = "1" ]; then
    lmabo_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete}" = "1" ]; then
    lmabo_args+=(--replace-incomplete-last-trial)
  fi

  OPENAI_API_KEY="${LLM_API_KEY}" \
  DASHSCOPE_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_KEY="${LLM_API_KEY}" \
  KIMI_MODEL="${LMABO_LLM_MODEL}" \
  LMABO_LLM_MODEL="${LMABO_LLM_MODEL}" \
  KIMI_BASE_URL="${DEEPSEEK_API_BASE}" \
  conda run --no-capture-output -n "${LMABO_CONDA_ENV}" "${lmabo_args[@]}"
}

run_dataset() {
  local ds="$1"
  echo "[Formal-DS][${ds^^}] Starting"
  if [ "${RUN_CAKE}" = "1" ]; then
    run_cake_dataset "${ds}"
  fi
  if [ "${RUN_LMABO}" = "1" ]; then
    run_lmabo_dataset "${ds}"
  fi
}

print_config

for ds in ${DATASETS}; do
  run_dataset "${ds}"
done
