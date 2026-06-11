#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-outputs/hpobench_bora}"
PROBLEMS="${PROBLEMS:-hpobench_rf_146606 hpobench_svm_146212 hpobench_xgb_146606 hpobench_nn_168912}"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"
TRIAL_START_INDEX="${TRIAL_START_INDEX:-1}"
SEED_START="${SEED_START:-0}"
TARGET_COLUMN="${TARGET_COLUMN:-test_acc}"

BORA_CONDA_ENV="${BORA_CONDA_ENV:-bora}"
DEEPSEEK_MODEL_NAME="${DEEPSEEK_MODEL_NAME:-deepseek-v4-pro}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
LLM_API_KEY="${LLM_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RERUN_FAILED_TRIALS="${RERUN_FAILED_TRIALS:-0}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

print_config() {
  echo "[HPOBench-BORA] problems=${PROBLEMS}"
  echo "[HPOBench-BORA] out_root=${OUT_ROOT} trials=${TARGET_TRIALS} budget=${TOTAL_BUDGET} init_size=${INIT_SIZE}"
  echo "[HPOBench-BORA] llm_model=${DEEPSEEK_MODEL_NAME} api_base=${DEEPSEEK_API_BASE}"
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
    echo "[HPOBench-BORA][${problem}] Missing dataset file: ${data_path}" >&2
    return 1
  fi
}

trial_output_dir() {
  local problem="$1"
  local trial_number="$2"
  printf '%s/%s/trial_%02d_run' "${OUT_ROOT}" "${problem}" "${trial_number}"
}

run_one_trial() {
  local problem="$1"
  local trial_number="$2"
  local seed="$3"
  local run_dir
  run_dir="$(trial_output_dir "${problem}" "${trial_number}")"

  if [ "${RERUN_FAILED_TRIALS}" = "1" ] && [ -d "${run_dir}" ] && [ ! -f "${run_dir}/${problem}_bora_results.npz" ]; then
    echo "[BORA][${problem}][trial_${trial_number}] Removing previous incomplete run dir ${run_dir}"
    rm -rf "${run_dir}"
  fi

  if [ -d "${run_dir}" ] && [ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    if [ -f "${run_dir}/${problem}_bora_results.npz" ]; then
      echo "[BORA][${problem}][trial_${trial_number}] Existing run found; skipping ${run_dir}"
      return 0
    fi
    echo "[BORA][${problem}][trial_${trial_number}] Non-empty incomplete run dir exists: ${run_dir}" >&2
    echo "[BORA][${problem}][trial_${trial_number}] Set RERUN_FAILED_TRIALS=1 to remove and retry it." >&2
    return 1
  fi

  echo "[BORA][${problem}][trial_${trial_number}] Starting seed=${seed}"
  OPENAI_API_KEY="${LLM_API_KEY}" \
  DASHSCOPE_API_KEY="${LLM_API_KEY}" \
  BORA_API_KEY="${LLM_API_KEY}" \
  DEEPSEEK_API_KEY="${LLM_API_KEY}" \
  BORA_LLM_MODEL="${DEEPSEEK_MODEL_NAME}" \
  KIMI_MODEL="${DEEPSEEK_MODEL_NAME}" \
  BORA_BASE_URL="${DEEPSEEK_API_BASE}" \
  KIMI_BASE_URL="${DEEPSEEK_API_BASE}" \
  conda run --no-capture-output -n "${BORA_CONDA_ENV}" python baseline/bora/run_tabular_bora.py \
    --dataset "${problem}" \
    --data-path "$(problem_data_path "${problem}")" \
    --target-column "${TARGET_COLUMN}" \
    --trials 1 \
    --trial-start-index "${trial_number}" \
    --seed-start "${seed}" \
    --total-budget "${TOTAL_BUDGET}" \
    --init-size "${INIT_SIZE}" \
    --output-dir "${run_dir}"
}

aggregate_problem() {
  local problem="$1"
  python -c "from pathlib import Path; import json, numpy as np, pandas as pd, sys
problem = sys.argv[1]
out_root = Path(sys.argv[2])
target_trials = int(sys.argv[3])
total_budget = int(sys.argv[4])
init_size = int(sys.argv[5])
target_column = sys.argv[6]
trial_start_index = int(sys.argv[7])
llm_model = sys.argv[8]
problem_dir = out_root / problem
run_dirs = [problem_dir / f'trial_{idx:02d}_run' for idx in range(trial_start_index, trial_start_index + target_trials)]
missing = [str(run_dir) for run_dir in run_dirs if not (run_dir / f'{problem}_bora_results.npz').exists()]
if missing:
    raise SystemExit('Missing per-trial result(s): ' + ', '.join(missing))

results_rows = []
trace_lengths = []
trial_numbers = []
observed_rows = []
best_rows = []
selected_rows = []
phase_rows = []
parameter_records_rows = []
parameter_columns = None

for run_dir in run_dirs:
    with np.load(run_dir / f'{problem}_bora_results.npz') as payload:
        results_rows.append(np.asarray(payload['results'], dtype=float)[0])
        trace_lengths.append(int(np.asarray(payload['trace_lengths'], dtype=int)[0]))
        trial_numbers.append(int(np.asarray(payload['trial_numbers'], dtype=int)[0]))
    with np.load(run_dir / f'{problem}_bora_trajectory.npz') as payload:
        observed_rows.append(np.asarray(payload['observed_values'], dtype=float)[0])
        best_rows.append(np.asarray(payload['best_values'], dtype=float)[0])
        selected_rows.append(np.asarray(payload['selected_indices'], dtype=int)[0])
        phase_rows.append(np.asarray(payload['phases'], dtype=str)[0])
        if 'parameter_columns' in payload.files and parameter_columns is None:
            parameter_columns = np.asarray(payload['parameter_columns'], dtype=str)
        if 'parameter_records' in payload.files:
            parameter_records_rows.append(np.asarray(payload['parameter_records'], dtype=str)[0])

width = max(len(row) for row in results_rows)
def pad_float(rows, fill=np.nan):
    matrix = np.full((len(rows), width), fill, dtype=float)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, :len(row)] = row
    return matrix
def pad_int(rows, fill=-1):
    matrix = np.full((len(rows), width), fill, dtype=int)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, :len(row)] = row
    return matrix
def pad_str(rows, fill=''):
    matrix = np.full((len(rows), width), fill, dtype=str)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, :len(row)] = row
    return matrix

results = pad_float(results_rows)
trace_lengths_array = np.asarray(trace_lengths, dtype=int)
trial_numbers_array = np.asarray(trial_numbers, dtype=int)
problem_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    problem_dir / f'{problem}_bora_results.npz',
    results=results,
    trace_lengths=trace_lengths_array,
    trial_numbers=trial_numbers_array,
)

trajectory_payload = {
    'observed_values': pad_float(observed_rows),
    'best_values': pad_float(best_rows),
    'selected_indices': pad_int(selected_rows),
    'phases': pad_str(phase_rows),
    'trace_lengths': trace_lengths_array,
    'trial_numbers': trial_numbers_array,
}
if parameter_columns is not None:
    trajectory_payload['parameter_columns'] = parameter_columns
if len(parameter_records_rows) == len(run_dirs):
    trajectory_payload['parameter_records'] = pad_str(parameter_records_rows)
np.savez(problem_dir / f'{problem}_bora_trajectory.npz', **trajectory_payload)

csv_rows = []
data_path = Path('data') / 'HPOBench' / f'{problem}.csv'
df = pd.read_csv(data_path)
for row_idx, trial_number in enumerate(trial_numbers_array.tolist()):
    for eval_idx in range(trace_lengths[row_idx]):
        dataset_index = int(trajectory_payload['selected_indices'][row_idx, eval_idx])
        row = {
            'trial_number': int(trial_number),
            'evaluation_index': int(eval_idx + 1),
            'dataset_index': dataset_index,
            'phase': str(trajectory_payload['phases'][row_idx, eval_idx]),
            'observed_objective': float(trajectory_payload['observed_values'][row_idx, eval_idx]),
            'best_objective': float(trajectory_payload['best_values'][row_idx, eval_idx]),
        }
        if 0 <= dataset_index < len(df):
            feature_payload = df.drop(columns=[target_column], errors='ignore').iloc[dataset_index].to_dict()
            row.update(feature_payload)
        csv_rows.append(row)
pd.DataFrame(csv_rows).to_csv(problem_dir / f'{problem}_bora_trajectory.csv', index=False)

initial_values = results[:, 0]
final_values = np.asarray([results[idx, length - 1] for idx, length in enumerate(trace_lengths)], dtype=float)
summary = {
    'dataset': problem,
    'trials': int(len(trial_numbers)),
    'trial_numbers': trial_numbers_array.tolist(),
    'total_budget': int(total_budget),
    'init_size': int(init_size),
    'llm_model': llm_model,
    'target_column': target_column,
    'assistant_mode': 'original_bora_control_flow',
    'search_space': 'semantic_discrete_with_validity_constraint',
    'initial_mean': float(np.mean(initial_values)),
    'final_mean': float(np.mean(final_values)),
    'initial_values': initial_values.tolist(),
    'final_values': final_values.tolist(),
}
(problem_dir / f'{problem}_bora_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'[BORA][{problem}] Aggregated {len(trial_numbers)} trial(s): final_mean={summary[\"final_mean\"]:.4f}')
" "${problem}" "${OUT_ROOT}" "${TARGET_TRIALS}" "${TOTAL_BUDGET}" "${INIT_SIZE}" "${TARGET_COLUMN}" "${TRIAL_START_INDEX}" "${DEEPSEEK_MODEL_NAME}"
}

run_problem() {
  local problem="$1"
  ensure_problem_exists "${problem}"
  mkdir -p "${OUT_ROOT}/${problem}"

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -f "${OUT_ROOT}/${problem}/${problem}_bora_results.npz" ] && [ -f "${OUT_ROOT}/${problem}/${problem}_bora_summary.json" ]; then
    echo "[BORA][${problem}] Aggregated outputs already exist; skipping."
    return 0
  fi

  echo "[HPOBench-BORA][${problem}] Starting ${TARGET_TRIALS} parallel trial(s)"
  pids=()
  for offset in $(seq 0 $((TARGET_TRIALS - 1))); do
    trial_number=$((TRIAL_START_INDEX + offset))
    seed=$((SEED_START + offset))
    run_one_trial "${problem}" "${trial_number}" "${seed}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    echo "[HPOBench-BORA][${problem}] At least one parallel trial failed." >&2
    return "${status}"
  fi

  aggregate_problem "${problem}"
}

print_config

for problem in ${PROBLEMS}; do
  run_problem "${problem}"
done
