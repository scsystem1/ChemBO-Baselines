#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-/home/sjtu/LLMBO/ChemBO-Baselines/outputs/bora}"
HPO_PROBLEMS="${HPO_PROBLEMS:-hpobench_rf_146606 hpobench_svm_146212 hpobench_xgb_146606 hpobench_nn_168912}"
CHEM_DATASETS="${CHEM_DATASETS:-dar ocm oer suzuki}"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"
TRIAL_START_INDEX="${TRIAL_START_INDEX:-1}"
SEED_START="${SEED_START:-0}"
HPO_TARGET_COLUMN="${HPO_TARGET_COLUMN:-test_acc}"

BORA_CONDA_ENV="${BORA_CONDA_ENV:-bora}"
DEEPSEEK_MODEL_NAME="${DEEPSEEK_MODEL_NAME:-deepseek-v4-pro}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
LLM_API_KEY="${LLM_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RERUN_FAILED_TRIALS="${RERUN_FAILED_TRIALS:-0}"
RERUN_EXISTING_TRIALS="${RERUN_EXISTING_TRIALS:-0}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

print_config() {
  echo "[BORA-Suite] hpo_problems=${HPO_PROBLEMS}"
  echo "[BORA-Suite] chem_datasets=${CHEM_DATASETS}"
  echo "[BORA-Suite] out_root=${OUT_ROOT} trials=${TARGET_TRIALS} budget=${TOTAL_BUDGET} init_size=${INIT_SIZE}"
  echo "[BORA-Suite] llm_model=${DEEPSEEK_MODEL_NAME} api_base=${DEEPSEEK_API_BASE}"
}

hpo_data_path() {
  local dataset="$1"
  printf '%s/data/HPOBench/%s.csv' "${ROOT_DIR}" "${dataset}"
}

is_hpo_dataset() {
  local dataset="$1"
  [[ " ${HPO_PROBLEMS} " == *" ${dataset} "* ]]
}

ensure_dataset_exists() {
  local dataset="$1"
  if is_hpo_dataset "${dataset}"; then
    local data_path
    data_path="$(hpo_data_path "${dataset}")"
    if [ ! -f "${data_path}" ]; then
      echo "[BORA][${dataset}] Missing dataset file: ${data_path}" >&2
      return 1
    fi
  fi
}

trial_output_dir() {
  local dataset="$1"
  local trial_number="$2"
  printf '%s/%s/trial_%02d_run' "${OUT_ROOT}" "${dataset}" "${trial_number}"
}

build_dataset_args() {
  local dataset="$1"
  if is_hpo_dataset "${dataset}"; then
    printf '%s
' --data-path "$(hpo_data_path "${dataset}")" --target-column "${HPO_TARGET_COLUMN}"
  fi
}

run_one_trial() {
  local dataset="$1"
  local trial_number="$2"
  local seed="$3"
  local run_dir
  run_dir="$(trial_output_dir "${dataset}" "${trial_number}")"

  if [ "${RERUN_EXISTING_TRIALS}" = "1" ] && [ -d "${run_dir}" ]; then
    echo "[BORA][${dataset}][trial_${trial_number}] Removing existing run dir ${run_dir}"
    rm -rf "${run_dir}"
  elif [ "${RERUN_FAILED_TRIALS}" = "1" ] && [ -d "${run_dir}" ] && [ ! -f "${run_dir}/${dataset}_bora_results.npz" ]; then
    echo "[BORA][${dataset}][trial_${trial_number}] Removing previous incomplete run dir ${run_dir}"
    rm -rf "${run_dir}"
  fi

  if [ -d "${run_dir}" ] && [ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    if [ -f "${run_dir}/${dataset}_bora_results.npz" ]; then
      echo "[BORA][${dataset}][trial_${trial_number}] Existing run found; skipping ${run_dir}"
      return 0
    fi
    echo "[BORA][${dataset}][trial_${trial_number}] Non-empty incomplete run dir exists: ${run_dir}" >&2
    echo "[BORA][${dataset}][trial_${trial_number}] Set RERUN_FAILED_TRIALS=1 to remove and retry it." >&2
    return 1
  fi

  local dataset_args=()
  while IFS= read -r arg; do
    dataset_args+=("${arg}")
  done < <(build_dataset_args "${dataset}")

  echo "[BORA][${dataset}][trial_${trial_number}] Starting seed=${seed}"
  OPENAI_API_KEY="${LLM_API_KEY}" \
  DASHSCOPE_API_KEY="${LLM_API_KEY}" \
  BORA_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_KEY="${LLM_API_KEY}" \
  BORA_LLM_MODEL="${DEEPSEEK_MODEL_NAME}" \
  KIMI_MODEL="${DEEPSEEK_MODEL_NAME}" \
  BORA_BASE_URL="${DEEPSEEK_API_BASE}" \
  KIMI_BASE_URL="${DEEPSEEK_API_BASE}" \
  conda run --no-capture-output -n "${BORA_CONDA_ENV}" python baseline/bora/run_tabular_bora.py \
    --dataset "${dataset}" \
    "${dataset_args[@]}" \
    --trials 1 \
    --trial-start-index "${trial_number}" \
    --seed-start "${seed}" \
    --total-budget "${TOTAL_BUDGET}" \
    --init-size "${INIT_SIZE}" \
    --output-dir "${run_dir}"
}

aggregate_dataset() {
  local dataset="$1"
  local aggregate_args=(
    python scripts/aggregate_bora_parallel_runs.py
    --dataset "${dataset}"
    --out-root "${OUT_ROOT}"
    --trials "${TARGET_TRIALS}"
    --trial-start-index "${TRIAL_START_INDEX}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --llm-model "${DEEPSEEK_MODEL_NAME}"
  )
  if is_hpo_dataset "${dataset}"; then
    aggregate_args+=(--data-path "$(hpo_data_path "${dataset}")" --target-column "${HPO_TARGET_COLUMN}")
  fi
  "${aggregate_args[@]}"
}

run_dataset() {
  local dataset="$1"
  ensure_dataset_exists "${dataset}"
  mkdir -p "${OUT_ROOT}/${dataset}"

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${OUT_ROOT}/${dataset}/${dataset}_bora_results.npz" ] && [ -f "${OUT_ROOT}/${dataset}/${dataset}_bora_summary.json" ]; then
    echo "[BORA][${dataset}] Aggregated outputs already exist; skipping."
    return 0
  fi

  echo "[BORA-Suite][${dataset}] Starting ${TARGET_TRIALS} parallel trial(s)"
  pids=()
  for offset in $(seq 0 $((TARGET_TRIALS - 1))); do
    trial_number=$((TRIAL_START_INDEX + offset))
    seed=$((SEED_START + offset))
    run_one_trial "${dataset}" "${trial_number}" "${seed}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    echo "[BORA-Suite][${dataset}] At least one parallel trial failed." >&2
    return "${status}"
  fi

  aggregate_dataset "${dataset}"
}

print_config

echo "[BORA-Suite] Running HPOBench tasks first"
for dataset in ${HPO_PROBLEMS}; do
  run_dataset "${dataset}"
done

echo "[BORA-Suite] Running chemistry tasks next"
for dataset in ${CHEM_DATASETS}; do
  run_dataset "${dataset}"
done
