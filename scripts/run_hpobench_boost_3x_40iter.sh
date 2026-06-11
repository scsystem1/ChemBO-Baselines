#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-/home/sjtu/LLMBO/ChemBO-Baselines/outputs/formal_runs_ds}"
PROBLEMS="${PROBLEMS:-hpobench_rf_146606 hpobench_svm_146212 hpobench_xgb_146606 hpobench_nn_168912}"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"
TARGET_COLUMN="${TARGET_COLUMN:-test_acc}"

BOOST_CONDA_ENV="${BOOST_CONDA_ENV:-boost}"
BOOST_DEVICE="${BOOST_DEVICE:-cpu}"
BOOST_OUTPUT_SUBDIR="${BOOST_OUTPUT_SUBDIR:-boost}"

print_config() {
  echo "[HPOBench-BOOST] problems=${PROBLEMS}"
  echo "[HPOBench-BOOST] out_root=${OUT_ROOT} trials=${TARGET_TRIALS} budget=${TOTAL_BUDGET} init_size=${INIT_SIZE}"
  echo "[HPOBench-BOOST] conda_env=${BOOST_CONDA_ENV} device=${BOOST_DEVICE}"
  echo "[HPOBench-BOOST] target_column=${TARGET_COLUMN}"
  echo "[HPOBench-BOOST] boost_dir=${OUT_ROOT}/${BOOST_OUTPUT_SUBDIR}/<dataset>"
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
    echo "[HPOBench-BOOST][${problem}] Missing dataset file: ${data_path}" >&2
    return 1
  fi
}

get_resume_info() {
  local dataset="$1"
  python -c "from pathlib import Path; import numpy as np, sys
out_root = Path(sys.argv[1])
output_subdir = sys.argv[2]
dataset = sys.argv[3]
total_budget = int(sys.argv[4])
results_path = out_root / output_subdir / dataset / f'{dataset}_boost_results.npz'
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
" "${OUT_ROOT}" "${BOOST_OUTPUT_SUBDIR}" "${dataset}" "${TOTAL_BUDGET}"
}

run_problem() {
  local problem="$1"
  ensure_problem_exists "${problem}"

  local resume_info existing_completed existing_recorded has_incomplete remaining_trials
  resume_info="$(get_resume_info "${problem}")"
  read -r existing_completed existing_recorded has_incomplete <<<"${resume_info}"
  remaining_trials=$((TARGET_TRIALS - existing_completed))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[BOOST][${problem}] Existing completed trials=${existing_completed}; nothing to run."
    return
  fi

  local trial_start_index seed_start data_path
  trial_start_index=$((existing_completed + 1))
  seed_start="${existing_completed}"
  data_path="$(problem_data_path "${problem}")"
  echo "[BOOST][${problem}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  local boost_args=(
    python baseline/boost/run_tabular_boost.py
    --dataset "${problem}"
    --data-path "${data_path}"
    --target-column "${TARGET_COLUMN}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --device "${BOOST_DEVICE}"
    --output-dir "${OUT_ROOT}/${BOOST_OUTPUT_SUBDIR}/${problem}"
  )

  if [ "${existing_recorded}" -gt 0 ] || [ "${has_incomplete}" = "1" ]; then
    boost_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete}" = "1" ]; then
    boost_args+=(--replace-incomplete-last-trial)
  fi

  conda run --no-capture-output -n "${BOOST_CONDA_ENV}" "${boost_args[@]}"
}

print_config

for problem in ${PROBLEMS}; do
  run_problem "${problem}"
done
