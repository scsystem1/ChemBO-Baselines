#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIALS_PER_DATASET="${TRIALS_PER_DATASET:-5}"
BUDGET="${BUDGET:-40}"
MAX_CPU_CORES="${MAX_CPU_CORES:-50}"
DAR_SEED_START="${DAR_SEED_START:-0}"
OCM_SEED_START="${OCM_SEED_START:-1000}"

if ! [[ "${TRIALS_PER_DATASET}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS_PER_DATASET must be a positive integer, got: ${TRIALS_PER_DATASET}" >&2
  exit 1
fi

if ! [[ "${BUDGET}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BUDGET must be a positive integer, got: ${BUDGET}" >&2
  exit 1
fi

if ! [[ "${MAX_CPU_CORES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CPU_CORES must be a positive integer, got: ${MAX_CPU_CORES}" >&2
  exit 1
fi

if ! [[ "${DAR_SEED_START}" =~ ^[0-9]+$ ]]; then
  echo "DAR_SEED_START must be a non-negative integer, got: ${DAR_SEED_START}" >&2
  exit 1
fi

if ! [[ "${OCM_SEED_START}" =~ ^[0-9]+$ ]]; then
  echo "OCM_SEED_START must be a non-negative integer, got: ${OCM_SEED_START}" >&2
  exit 1
fi

echo "[PrefBO][Suite] Running DAR first, then OCM. trials_per_dataset=${TRIALS_PER_DATASET}, budget=${BUDGET}, max_cpu_cores=${MAX_CPU_CORES}"
echo "[PrefBO][Suite] DAR seeds: ${DAR_SEED_START}-$((DAR_SEED_START + TRIALS_PER_DATASET - 1))"
echo "[PrefBO][Suite] OCM seeds: ${OCM_SEED_START}-$((OCM_SEED_START + TRIALS_PER_DATASET - 1))"

TRIALS="${TRIALS_PER_DATASET}" \
BUDGET="${BUDGET}" \
SEED_START="${DAR_SEED_START}" \
MAX_CPU_CORES="${MAX_CPU_CORES}" \
bash "${ROOT_DIR}/scripts/run_prefbo_dar_5x_40iter.sh"

TRIALS="${TRIALS_PER_DATASET}" \
BUDGET="${BUDGET}" \
SEED_START="${OCM_SEED_START}" \
MAX_CPU_CORES="${MAX_CPU_CORES}" \
bash "${ROOT_DIR}/scripts/run_prefbo_ocm_5x_40iter.sh"
