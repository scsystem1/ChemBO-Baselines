#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-oer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/formal_runs/bora/${DATASET}}"
SEED_START="${SEED_START:-0}"
TRIALS="${TRIALS:-3}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"
TRIAL_START_INDEX="${TRIAL_START_INDEX:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
BACKUP_INCOMPLETE="${BACKUP_INCOMPLETE:-1}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${OUTPUT_DIR}"

RESULTS_PATH="${OUTPUT_DIR}/${DATASET}_bora_results.npz"
SUMMARY_PATH="${OUTPUT_DIR}/${DATASET}_bora_summary.json"

if [[ "${SKIP_COMPLETED}" == "1" && -f "${RESULTS_PATH}" && -f "${SUMMARY_PATH}" ]]; then
  echo "[BORA][${DATASET^^}] Skipping because completed outputs already exist at ${OUTPUT_DIR}"
  exit 0
fi

if [[ "${BACKUP_INCOMPLETE}" == "1" && -d "${OUTPUT_DIR}" && ! -f "${RESULTS_PATH}" ]]; then
  BACKUP_DIR="${OUTPUT_DIR}_incomplete_$(date +%Y%m%d_%H%M%S)"
  echo "[BORA][${DATASET^^}] Found incomplete outputs. Moving them to ${BACKUP_DIR}"
  mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
  mkdir -p "${OUTPUT_DIR}"
fi

echo "[BORA][${DATASET^^}] Starting run with budget=${TOTAL_BUDGET}, init_size=${INIT_SIZE}, trials=${TRIALS}, seed_start=${SEED_START}"
conda run --no-capture-output -n bora python "${ROOT_DIR}/baseline/bora/run_tabular_bora.py" \
  --dataset "${DATASET}" \
  --trials "${TRIALS}" \
  --trial-start-index "${TRIAL_START_INDEX}" \
  --seed-start "${SEED_START}" \
  --total-budget "${TOTAL_BUDGET}" \
  --init-size "${INIT_SIZE}" \
  --output-dir "${OUTPUT_DIR}"

echo "[BORA][${DATASET^^}] Completed. Outputs are under ${OUTPUT_DIR}"
