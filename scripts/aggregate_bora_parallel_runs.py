#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "baseline"))

from common.tabular_benchmarks import load_benchmark_spec


def _pad_float(rows: list[np.ndarray], width: int, fill: float = np.nan) -> np.ndarray:
    matrix = np.full((len(rows), width), fill, dtype=float)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, : len(row)] = row
    return matrix


def _pad_int(rows: list[np.ndarray], width: int, fill: int = -1) -> np.ndarray:
    matrix = np.full((len(rows), width), fill, dtype=int)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, : len(row)] = row
    return matrix


def _pad_str(rows: list[np.ndarray], width: int, fill: str = "") -> np.ndarray:
    matrix = np.full((len(rows), width), fill, dtype=str)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, : len(row)] = row
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate parallel one-trial BORA runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--trial-start-index", type=int, default=1)
    parser.add_argument("--total-budget", type=int, required=True)
    parser.add_argument("--init-size", type=int, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target-column", default=None)
    args = parser.parse_args()

    spec, df = load_benchmark_spec(
        ROOT,
        args.dataset,
        data_path=args.data_path,
        target_column=args.target_column,
    )

    out_root = Path(args.out_root)
    dataset_dir = out_root / args.dataset
    run_dirs = [
        dataset_dir / f"trial_{idx:02d}_run"
        for idx in range(args.trial_start_index, args.trial_start_index + args.trials)
    ]
    missing = [str(run_dir) for run_dir in run_dirs if not (run_dir / f"{args.dataset}_bora_results.npz").exists()]
    if missing:
        raise SystemExit("Missing per-trial result(s): " + ", ".join(missing))

    results_rows: list[np.ndarray] = []
    trace_lengths: list[int] = []
    trial_numbers: list[int] = []
    observed_rows: list[np.ndarray] = []
    best_rows: list[np.ndarray] = []
    selected_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    parameter_records_rows: list[np.ndarray] = []
    parameter_columns: np.ndarray | None = None

    for run_dir in run_dirs:
        with np.load(run_dir / f"{args.dataset}_bora_results.npz") as payload:
            results_rows.append(np.asarray(payload["results"], dtype=float)[0])
            trace_lengths.append(int(np.asarray(payload["trace_lengths"], dtype=int)[0]))
            trial_numbers.append(int(np.asarray(payload["trial_numbers"], dtype=int)[0]))
        with np.load(run_dir / f"{args.dataset}_bora_trajectory.npz") as payload:
            observed_rows.append(np.asarray(payload["observed_values"], dtype=float)[0])
            best_rows.append(np.asarray(payload["best_values"], dtype=float)[0])
            selected_rows.append(np.asarray(payload["selected_indices"], dtype=int)[0])
            phase_rows.append(np.asarray(payload["phases"], dtype=str)[0])
            if "parameter_columns" in payload.files and parameter_columns is None:
                parameter_columns = np.asarray(payload["parameter_columns"], dtype=str)
            if "parameter_records" in payload.files:
                parameter_records_rows.append(np.asarray(payload["parameter_records"], dtype=str)[0])

    width = max(len(row) for row in results_rows)
    results = _pad_float(results_rows, width)
    trace_lengths_array = np.asarray(trace_lengths, dtype=int)
    trial_numbers_array = np.asarray(trial_numbers, dtype=int)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        dataset_dir / f"{args.dataset}_bora_results.npz",
        results=results,
        trace_lengths=trace_lengths_array,
        trial_numbers=trial_numbers_array,
    )

    trajectory_payload: dict[str, object] = {
        "observed_values": _pad_float(observed_rows, width),
        "best_values": _pad_float(best_rows, width),
        "selected_indices": _pad_int(selected_rows, width),
        "phases": _pad_str(phase_rows, width),
        "trace_lengths": trace_lengths_array,
        "trial_numbers": trial_numbers_array,
    }
    if parameter_columns is not None:
        trajectory_payload["parameter_columns"] = parameter_columns
    if len(parameter_records_rows) == len(run_dirs):
        trajectory_payload["parameter_records"] = _pad_str(parameter_records_rows, width)
    np.savez(dataset_dir / f"{args.dataset}_bora_trajectory.npz", **trajectory_payload)

    parameter_df = df.drop(columns=[spec.target_column], errors="ignore").reset_index(drop=True)
    csv_rows: list[dict[str, object]] = []
    selected_matrix = trajectory_payload["selected_indices"]
    observed_matrix = trajectory_payload["observed_values"]
    best_matrix = trajectory_payload["best_values"]
    phase_matrix = trajectory_payload["phases"]
    for row_idx, trial_number in enumerate(trial_numbers_array.tolist()):
        for eval_idx in range(trace_lengths[row_idx]):
            dataset_index = int(selected_matrix[row_idx, eval_idx])
            row: dict[str, object] = {
                "trial_number": int(trial_number),
                "evaluation_index": int(eval_idx + 1),
                "dataset_index": dataset_index,
                "phase": str(phase_matrix[row_idx, eval_idx]),
                "observed_objective": float(observed_matrix[row_idx, eval_idx]),
                "best_objective": float(best_matrix[row_idx, eval_idx]),
            }
            if 0 <= dataset_index < len(parameter_df):
                row.update(parameter_df.iloc[dataset_index].to_dict())
            csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(dataset_dir / f"{args.dataset}_bora_trajectory.csv", index=False)

    initial_values = results[:, 0]
    final_values = np.asarray(
        [results[idx, length - 1] for idx, length in enumerate(trace_lengths)],
        dtype=float,
    )
    summary = {
        "dataset": args.dataset,
        "trials": int(len(trial_numbers)),
        "trial_numbers": trial_numbers_array.tolist(),
        "total_budget": int(args.total_budget),
        "init_size": int(args.init_size),
        "llm_model": args.llm_model,
        "target_column": spec.target_column,
        "assistant_mode": "original_bora_control_flow",
        "search_space": "semantic_discrete_with_validity_constraint",
        "feature_columns": spec.feature_columns,
        "initial_mean": float(np.mean(initial_values)),
        "final_mean": float(np.mean(final_values)),
        "initial_values": initial_values.tolist(),
        "final_values": final_values.tolist(),
    }
    (dataset_dir / f"{args.dataset}_bora_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[BORA][{args.dataset}] Aggregated {len(trial_numbers)} trial(s): "
        f"final_mean={summary['final_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
