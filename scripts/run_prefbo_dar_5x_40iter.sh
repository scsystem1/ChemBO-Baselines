#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIALS="${TRIALS:-5}"
BUDGET="${BUDGET:-40}"
SEED_START="${SEED_START:-0}"
MAX_CPU_CORES="${MAX_CPU_CORES:-50}"
PREFBO_OUTPUT_DIR="${PREFBO_OUTPUT_DIR:-${ROOT_DIR}/outputs/baseline_runs/prefbo/dar}"
PREFBO_QUESTIONS_FILE="${PREFBO_QUESTIONS_FILE:-${ROOT_DIR}/baseline/Pref-BO/questions/dar_questions_seed0.npy}"
PREFBO_MAX_QUESTIONS="${PREFBO_MAX_QUESTIONS:-2000}"

export PREFBO_LLM_MODEL="${PREFBO_LLM_MODEL:-kimi-k2.5-thinking}"
export PREFBO_BASE_URL="${PREFBO_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if ! [[ "${TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS must be a positive integer, got: ${TRIALS}" >&2
  exit 1
fi

if ! [[ "${BUDGET}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUDGET must be a positive integer, got: ${BUDGET}" >&2
  exit 1
fi

if ! [[ "${SEED_START}" =~ ^[0-9]+$ ]]; then
  echo "SEED_START must be a non-negative integer, got: ${SEED_START}" >&2
  exit 1
fi

if ! [[ "${MAX_CPU_CORES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CPU_CORES must be a positive integer, got: ${MAX_CPU_CORES}" >&2
  exit 1
fi

if [[ ! -f "${PREFBO_QUESTIONS_FILE}" ]]; then
  echo "PrefBO questions file not found: ${PREFBO_QUESTIONS_FILE}" >&2
  exit 1
fi

if ! [[ "${PREFBO_MAX_QUESTIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFBO_MAX_QUESTIONS must be a positive integer, got: ${PREFBO_MAX_QUESTIONS}" >&2
  exit 1
fi

AVAILABLE_CORES=1
if command -v getconf >/dev/null 2>&1; then
  AVAILABLE_CORES="$(getconf _NPROCESSORS_ONLN)"
elif command -v nproc >/dev/null 2>&1; then
  AVAILABLE_CORES="$(nproc)"
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

echo "[PrefBO][DAR] Starting run with budget=${BUDGET}, trials=${TRIALS}, seed_start=${SEED_START}, seed_end=$((SEED_START + TRIALS - 1)), cpu_cap=${CPU_THREAD_CAP}, questions=${PREFBO_QUESTIONS_FILE}"
"${CPU_AFFINITY_PREFIX[@]}" conda run --no-capture-output -n prefbo python "${ROOT_DIR}/baseline/Pref-BO/run_tabular_preference_bo.py" \
  --dataset dar \
  --trials "${TRIALS}" \
  --seed-start "${SEED_START}" \
  --total-budget "${BUDGET}" \
  --questions-file "${PREFBO_QUESTIONS_FILE}" \
  --max-questions "${PREFBO_MAX_QUESTIONS}" \
  --max-cpu-threads "${CPU_THREAD_CAP}" \
  --output-dir "${PREFBO_OUTPUT_DIR}"
