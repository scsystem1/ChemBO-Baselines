#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_TOOL="${CONDA_ENV_TOOL:-conda}"

if ! command -v "${CONDA_ENV_TOOL}" >/dev/null 2>&1; then
  echo "${CONDA_ENV_TOOL} is required but was not found in PATH." >&2
  exit 1
fi

echo "[setup] Env tool=${CONDA_ENV_TOOL}"

env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -qx "${env_name}"
}

ensure_runner_env() {
  local env_name="${1:-CAS}"
  if env_exists "${env_name}"; then
    echo "[setup] Reusing runner env ${env_name}"
  else
    echo "[setup] Creating runner env ${env_name}"
    "${CONDA_ENV_TOOL}" create -y -n "${env_name}" python=3.10
  fi
}

apply_conda_env_file() {
  local env_name="$1"
  local env_file="$2"
  if [[ ! -f "${env_file}" ]]; then
    echo "[setup] Skip ${env_name}: missing ${env_file}"
    return 0
  fi
  if env_exists "${env_name}"; then
    echo "[setup] Updating ${env_name} from ${env_file}"
    "${CONDA_ENV_TOOL}" env update -n "${env_name}" -f "${env_file}" --prune
  else
    echo "[setup] Creating ${env_name} from ${env_file}"
    "${CONDA_ENV_TOOL}" env create -f "${env_file}"
  fi
}

ensure_runner_env "${RUNNER_CONDA_ENV:-CAS}"
apply_conda_env_file "prefbo" "${ROOT_DIR}/baseline/Pref-BO/environment.yml"
apply_conda_env_file "reasoning_bo" "${ROOT_DIR}/baseline/Reasoning-BO/environment.yml"
apply_conda_env_file "gollum" "${ROOT_DIR}/baseline/gollum/environment.yaml"
apply_conda_env_file "bora" "${ROOT_DIR}/baseline/bora/environment.yml"
apply_conda_env_file "lmabo" "${ROOT_DIR}/baseline/lmabo/environment.yml"
apply_conda_env_file "cake" "${ROOT_DIR}/baseline/cake/environment.yml"

if [[ -f "${ROOT_DIR}/baseline/BO-ICL/environment.yml" ]]; then
  apply_conda_env_file "boicl" "${ROOT_DIR}/baseline/BO-ICL/environment.yml"
else
  echo "[setup] Skip boicl: baseline/BO-ICL/environment.yml is not present in this worktree."
fi

echo "[setup] Baseline environment bootstrap complete."
