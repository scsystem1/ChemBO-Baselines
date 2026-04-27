#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIALS="${TRIALS:-3}"
BUDGET="${BUDGET:-40}"
SEED_START="${SEED_START:-0}"
MAX_CPU_CORES="${MAX_CPU_CORES:-50}"
export PREFBO_LLM_MODEL="${PREFBO_LLM_MODEL:-kimi-k2.5-thinking}"
export PREFBO_BASE_URL="${PREFBO_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if command -v getconf >/dev/null 2>&1; then
  AVAILABLE_CORES="$(getconf _NPROCESSORS_ONLN)"
elif command -v nproc >/dev/null 2>&1; then
  AVAILABLE_CORES="$(nproc)"
else
  AVAILABLE_CORES=1
fi

if [[ "${AVAILABLE_CORES}" =~ ^[1-9][0-9]*$ ]] && (( MAX_CPU_CORES > AVAILABLE_CORES )); then
  CPU_THREAD_CAP="${AVAILABLE_CORES}"
else
  CPU_THREAD_CAP="${MAX_CPU_CORES}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREAD_CAP}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREAD_CAP}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREAD_CAP}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREAD_CAP}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREAD_CAP}}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-${CPU_THREAD_CAP}}"

CPU_AFFINITY_PREFIX=()
if command -v taskset >/dev/null 2>&1; then
  CPU_AFFINITY_PREFIX=("taskset" "-c" "0-$((CPU_THREAD_CAP - 1))")
fi

echo "[PrefBO][OCM] Starting run with model=${PREFBO_LLM_MODEL}, budget=${BUDGET}, trials=${TRIALS}, seed_start=${SEED_START}, seed_end=$((SEED_START + TRIALS - 1)), cpu_cap=${CPU_THREAD_CAP}, survey_questions=first_20000_from_file"
"${CPU_AFFINITY_PREFIX[@]}" conda run --no-capture-output -n prefbo python "${ROOT_DIR}/baseline/Pref-BO/run_tabular_preference_bo.py" \
  --dataset ocm \
  --trials "${TRIALS}" \
  --seed-start "${SEED_START}" \
  --total-budget "${BUDGET}" \
  --questions-file "${ROOT_DIR}/baseline/Pref-BO/questions/ocm_questions_seed0.npy" \
  --max-questions 20000 \
  --max-cpu-threads "${CPU_THREAD_CAP}" \
  --output-dir "${ROOT_DIR}/outputs/baseline_runs/prefbo/ocm"
