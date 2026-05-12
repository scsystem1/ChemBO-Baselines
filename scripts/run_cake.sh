cd /home/sjtu/LLMBO/ChemBO-Baselines

export DASHSCOPE_API_KEY='sk-565d62c64d2740e69f3acdadc7f58974'
OUT_ROOT="outputs/formal_runs"
TARGET_TRIALS="${TARGET_TRIALS:-3}"
DATASETS="${DATASETS:- ocm }"
TOTAL_BUDGET="${TOTAL_BUDGET:-40}"
INIT_SIZE="${INIT_SIZE:-10}"
CAKE_MODEL="${CAKE_MODEL:-kimi-k2.5}"

for ds in ${DATASETS}; do
  resume_info="$(python -c "from pathlib import Path; import numpy as np, sys
ds = sys.argv[1]
out_root = Path(sys.argv[2])
results_path = out_root / 'cake_2' / ds / f'{ds}_cake_results.npz'
total_budget = int(sys.argv[3])
completed = 0
has_incomplete = 0
if results_path.exists():
    payload = np.load(results_path)
    trace_lengths = payload['trace_lengths'].astype(int).tolist()
    completed = sum(1 for trace_len in trace_lengths if trace_len >= total_budget)
    has_incomplete = int(any(trace_len < total_budget for trace_len in trace_lengths))
print(f'{completed} {has_incomplete}')
" "${ds}" "${OUT_ROOT}" "${TOTAL_BUDGET}")"
  existing_trials="${resume_info%% *}"
  has_incomplete_trial="${resume_info##* }"
  remaining_trials=$((TARGET_TRIALS - existing_trials))

  if [ "${remaining_trials}" -le 0 ]; then
    echo "[CAKE][${ds^^}] Existing trials=${existing_trials}; nothing to resume."
    continue
  fi

  trial_start_index=$((existing_trials + 1))
  seed_start=$((existing_trials + 42))
  echo "[CAKE][${ds^^}] Resuming from trial ${trial_start_index}; running ${remaining_trials} more trial(s)."

  cake_args=(
    python baseline/cake/run_tabular_cake.py
    --dataset "${ds}"
    --trials "${remaining_trials}"
    --trial-start-index "${trial_start_index}"
    --seed-start "${seed_start}"
    --total-budget "${TOTAL_BUDGET}"
    --init-size "${INIT_SIZE}"
    --enable-llm-evolution
    --model-name "${CAKE_MODEL}"
    --output-dir "${OUT_ROOT}/cake_2/${ds}"
  )

  if [ "${existing_trials}" -gt 0 ] || [ "${has_incomplete_trial}" = "1" ]; then
    cake_args+=(--append-to-existing)
  fi
  if [ "${has_incomplete_trial}" = "1" ]; then
    cake_args+=(--replace-incomplete-last-trial)
  fi

  KIMI_MODEL="${CAKE_MODEL}" \
  conda run --no-capture-output -n cake "${cake_args[@]}"
done
