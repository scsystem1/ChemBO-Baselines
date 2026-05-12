cd /home/sjtu/LLMBO/ChemBO-Baselines

export DASHSCOPE_API_KEY='sk-565d62c64d2740e69f3acdadc7f58974'
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="outputs/formal_runs"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
DATASETS="${DATASETS:-dar ocm suzuki}"
RUN_CAKE="${RUN_CAKE:-0}"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"

for ds in ${DATASETS}; do
#   conda run --no-capture-output -n hebo \
#     python baseline/hebo/run_tabular_hebo.py \
#     --dataset "${ds}" \
#     --trials 3 \
#     --trial-start-index 1 \
#     --seed-start 0 \
#     --total-budget 40 \
#     --init-size 10 \
#     --output-dir "${OUT_ROOT}/hebo/${ds}"

  resume_info="$(python -c "from pathlib import Path; import numpy as np, sys
ds = sys.argv[1]
out_root = Path(sys.argv[2])
results_path = out_root / 'lmabo' / ds / f'{ds}_lmabo_results.npz'
total_budget = int(sys.argv[3])
completed = 0
has_incomplete = 0
if results_path.exists():
    payload = np.load(results_path)
    trial_numbers = payload['trial_numbers'].astype(int).tolist()
    trace_lengths = payload['trace_lengths'].astype(int).tolist()
    completed = sum(1 for trace_len in trace_lengths if trace_len >= total_budget)
    has_incomplete = int(any(trace_len < total_budget for trace_len in trace_lengths))
print(f'{completed} {has_incomplete}')
" "${ds}" "${OUT_ROOT}" "${TOTAL_BUDGET}")"
  existing_trials="${resume_info%% *}"
  has_incomplete_trial="${resume_info##* }"
  remaining_trials=$((TARGET_TRIALS - existing_trials))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[LMABO][${ds^^}] Existing trials=${existing_trials}; nothing to resume."
    continue
  fi

  trial_start_index=$((existing_trials + 1))
  seed_start="${existing_trials}"
  echo "[LMABO][${ds^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  lmabo_args=(
    python baseline/lmabo/run_tabular_lmabo.py
    --dataset "${ds}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --llm-mode api
    --api-type gpt
    --conversation-init-retries 3
    --conversation-init-delay-seconds 5
    --output-dir "${OUT_ROOT}/lmabo/${ds}"
  )

  if [ "${existing_trials}" -gt 0 ] || [ "${has_incomplete_trial}" = "1" ]; then
    lmabo_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete_trial}" = "1" ]; then
    lmabo_args+=(--replace-incomplete-last-trial)
  fi

  KIMI_MODEL='kimi-k2.5' LMABO_LLM_MODEL='kimi-k2.5' \
  conda run --no-capture-output -n lmabo "${lmabo_args[@]}"

  if [ "${RUN_CAKE}" = "1" ]; then
    KIMI_MODEL='kimi-k2.5-thinking' \
    conda run --no-capture-output -n cake \
      python baseline/cake/run_tabular_cake.py \
      --dataset "${ds}" \
      --trials 3 \
      --trial-start-index 1 \
      --seed-start 0 \
      --total-budget "${TOTAL_BUDGET}" \
      --init-size "${INIT_SIZE}" \
      --enable-llm-evolution \
      --model-name kimi-k2.5-thinking \
      --output-dir "${OUT_ROOT}/cake/${ds}"
  fi
done
