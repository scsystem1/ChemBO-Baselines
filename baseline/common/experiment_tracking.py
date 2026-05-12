from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrialTrace:
    selected_indices: np.ndarray
    observed_values: np.ndarray
    best_values: np.ndarray
    phases: np.ndarray | None = None

    def __post_init__(self) -> None:
        selected_indices = np.asarray(self.selected_indices, dtype=int).reshape(-1)
        observed_values = np.asarray(self.observed_values, dtype=float).reshape(-1)
        best_values = np.asarray(self.best_values, dtype=float).reshape(-1)
        if not (
            len(selected_indices) == len(observed_values) == len(best_values)
        ):
            raise ValueError("TrialTrace arrays must have the same length.")
        if self.phases is not None and len(np.asarray(self.phases).reshape(-1)) != len(observed_values):
            raise ValueError("TrialTrace phases must align with the observation count.")

    @property
    def length(self) -> int:
        return int(len(np.asarray(self.observed_values).reshape(-1)))


def cumulative_best(values: Iterable[float]) -> np.ndarray:
    observed = np.asarray(list(values), dtype=float).reshape(-1)
    if observed.size == 0:
        return observed
    return np.maximum.accumulate(observed)


def build_trial_trace(
    selected_indices: Iterable[int],
    observed_values: Iterable[float],
    phases: Iterable[str] | None = None,
) -> TrialTrace:
    observed = np.asarray(list(observed_values), dtype=float).reshape(-1)
    selected = np.asarray(list(selected_indices), dtype=int).reshape(-1)
    if observed.size != selected.size:
        raise ValueError("selected_indices and observed_values must have the same length.")
    if phases is None:
        phase_values = None
    else:
        phase_values = np.asarray(list(phases), dtype=str).reshape(-1)
    return TrialTrace(
        selected_indices=selected,
        observed_values=observed,
        best_values=cumulative_best(observed),
        phases=phase_values,
    )


def _pad_float_sequences(sequences: list[np.ndarray], fill_value: float = np.nan) -> np.ndarray:
    max_len = max((len(sequence) for sequence in sequences), default=0)
    matrix = np.full((len(sequences), max_len), fill_value, dtype=float)
    for row_idx, sequence in enumerate(sequences):
        matrix[row_idx, : len(sequence)] = np.asarray(sequence, dtype=float)
    return matrix


def _pad_float_matrix(matrix: np.ndarray, target_width: int, fill_value: float = np.nan) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] >= target_width:
        return matrix
    padded = np.full((matrix.shape[0], target_width), fill_value, dtype=float)
    padded[:, : matrix.shape[1]] = matrix
    return padded


def _pad_int_sequences(sequences: list[np.ndarray], fill_value: int = -1) -> np.ndarray:
    max_len = max((len(sequence) for sequence in sequences), default=0)
    matrix = np.full((len(sequences), max_len), fill_value, dtype=int)
    for row_idx, sequence in enumerate(sequences):
        matrix[row_idx, : len(sequence)] = np.asarray(sequence, dtype=int)
    return matrix


def _pad_str_sequences(sequences: list[np.ndarray], fill_value: str = "") -> np.ndarray:
    max_len = max((len(sequence) for sequence in sequences), default=0)
    matrix = np.full((len(sequences), max_len), fill_value, dtype=str)
    for row_idx, sequence in enumerate(sequences):
        matrix[row_idx, : len(sequence)] = np.asarray(sequence, dtype=str)
    return matrix


def build_best_results_matrix(traces: list[TrialTrace]) -> tuple[np.ndarray, np.ndarray]:
    best_sequences = [np.asarray(trace.best_values, dtype=float) for trace in traces]
    trace_lengths = np.asarray([trace.length for trace in traces], dtype=int)
    return _pad_float_sequences(best_sequences), trace_lengths


def combine_best_results(
    traces: list[TrialTrace],
    trial_numbers: np.ndarray,
    *,
    existing_results: np.ndarray | None = None,
    existing_trace_lengths: np.ndarray | None = None,
    existing_trial_numbers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_results, new_trace_lengths = build_best_results_matrix(traces)
    current_trial_numbers = np.asarray(trial_numbers, dtype=int)
    prior_trial_numbers = (
        np.asarray(existing_trial_numbers, dtype=int)
        if existing_trial_numbers is not None
        else np.empty(0, dtype=int)
    )

    if existing_results is not None and existing_trace_lengths is not None:
        combined_width = max(existing_results.shape[1], new_results.shape[1])
        results = np.concatenate(
            [
                _pad_float_matrix(np.asarray(existing_results, dtype=float), combined_width),
                _pad_float_matrix(np.asarray(new_results, dtype=float), combined_width),
            ],
            axis=0,
        )
        trace_lengths = np.concatenate(
            [
                np.asarray(existing_trace_lengths, dtype=int),
                new_trace_lengths,
            ]
        )
        merged_trial_numbers = np.concatenate([prior_trial_numbers, current_trial_numbers])
        return results, trace_lengths, merged_trial_numbers

    return new_results, new_trace_lengths, current_trial_numbers


def summarize_best_results(results: np.ndarray, trace_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if results.shape[0] != trace_lengths.shape[0]:
        raise ValueError("results and trace_lengths must have the same number of rows.")
    initial_values = np.asarray(results[:, 0], dtype=float)
    final_values = np.asarray(
        [results[row_idx, int(trace_len) - 1] for row_idx, trace_len in enumerate(trace_lengths)],
        dtype=float,
    )
    return initial_values, final_values


def _build_parameter_frame(
    dataset_df: pd.DataFrame | None,
    target_column: str | None,
) -> pd.DataFrame | None:
    if dataset_df is None:
        return None
    parameter_df = dataset_df.copy()
    if target_column is not None and target_column in parameter_df.columns:
        parameter_df = parameter_df.drop(columns=[target_column])
    return parameter_df.reset_index(drop=True)


def _build_parameter_record_matrix(
    selected_index_matrix: np.ndarray,
    trace_lengths: np.ndarray,
    parameter_df: pd.DataFrame | None,
) -> np.ndarray | None:
    if parameter_df is None:
        return None
    record_rows = [["" for _ in range(selected_index_matrix.shape[1])] for _ in range(selected_index_matrix.shape[0])]
    for row_idx, trace_len in enumerate(trace_lengths.tolist()):
        for col_idx in range(int(trace_len)):
            dataset_index = int(selected_index_matrix[row_idx, col_idx])
            if dataset_index < 0 or dataset_index >= len(parameter_df):
                continue
            record = parameter_df.iloc[dataset_index].to_dict()
            record_rows[row_idx][col_idx] = json.dumps(record, ensure_ascii=False, default=str)
    return np.asarray(record_rows, dtype=str)


def export_trajectory_artifacts(
    output_dir: Path,
    stem: str,
    traces: list[TrialTrace],
    trial_numbers: np.ndarray,
    *,
    dataset_df: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_lengths = np.asarray([trace.length for trace in traces], dtype=int)
    observed_matrix = _pad_float_sequences(
        [np.asarray(trace.observed_values, dtype=float) for trace in traces]
    )
    best_matrix = _pad_float_sequences(
        [np.asarray(trace.best_values, dtype=float) for trace in traces]
    )
    selected_index_matrix = _pad_int_sequences(
        [np.asarray(trace.selected_indices, dtype=int) for trace in traces]
    )
    phase_matrix = _pad_str_sequences(
        [
            (
                np.asarray(trace.phases, dtype=str).reshape(-1)
                if trace.phases is not None
                else np.full(trace.length, "", dtype=str)
            )
            for trace in traces
        ]
    )
    parameter_df = _build_parameter_frame(dataset_df, target_column)
    parameter_record_matrix = _build_parameter_record_matrix(
        selected_index_matrix,
        trace_lengths,
        parameter_df,
    )

    npy_path = output_dir / f"{stem}_trajectory.npz"
    npz_payload: dict[str, object] = {
        "observed_values": observed_matrix,
        "best_values": best_matrix,
        "selected_indices": selected_index_matrix,
        "phases": phase_matrix,
        "trace_lengths": trace_lengths,
        "trial_numbers": np.asarray(trial_numbers, dtype=int),
    }
    if parameter_df is not None:
        npz_payload["parameter_columns"] = np.asarray(parameter_df.columns.tolist(), dtype=str)
    if parameter_record_matrix is not None:
        npz_payload["parameter_records"] = parameter_record_matrix
    np.savez(npy_path, **npz_payload)

    csv_rows: list[dict[str, object]] = []
    for trace, trial_number in zip(traces, trial_numbers.tolist()):
        phases = (
            np.asarray(trace.phases, dtype=str).reshape(-1)
            if trace.phases is not None
            else np.full(trace.length, "", dtype=str)
        )
        for eval_idx, (dataset_index, observed, best_value, phase) in enumerate(
            zip(
                np.asarray(trace.selected_indices, dtype=int).tolist(),
                np.asarray(trace.observed_values, dtype=float).tolist(),
                np.asarray(trace.best_values, dtype=float).tolist(),
                phases.tolist(),
            ),
            start=1,
        ):
            csv_row = {
                "trial_number": int(trial_number),
                "evaluation_index": int(eval_idx),
                "dataset_index": int(dataset_index),
                "phase": phase,
                "observed_objective": float(observed),
                "best_objective": float(best_value),
            }
            if parameter_df is not None and 0 <= int(dataset_index) < len(parameter_df):
                csv_row.update(parameter_df.iloc[int(dataset_index)].to_dict())
            csv_rows.append(
                csv_row
            )

    csv_path = output_dir / f"{stem}_trajectory.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    return npy_path, csv_path


def load_existing_trajectory_artifacts(
    trajectory_path: Path,
) -> tuple[list[TrialTrace], np.ndarray] | None:
    if not trajectory_path.exists():
        return None

    with np.load(trajectory_path) as payload:
        observed_matrix = np.asarray(payload["observed_values"], dtype=float)
        selected_index_matrix = np.asarray(payload["selected_indices"], dtype=int)
        phase_matrix = (
            np.asarray(payload["phases"], dtype=str)
            if "phases" in payload.files
            else None
        )
        if "trace_lengths" in payload.files:
            trace_lengths = np.asarray(payload["trace_lengths"], dtype=int)
        else:
            trace_lengths = np.sum(~np.isnan(observed_matrix), axis=1, dtype=int)
        if "trial_numbers" in payload.files:
            trial_numbers = np.asarray(payload["trial_numbers"], dtype=int)
        else:
            trial_numbers = np.arange(1, observed_matrix.shape[0] + 1, dtype=int)

    traces: list[TrialTrace] = []
    for row_idx, trace_len in enumerate(trace_lengths.tolist()):
        trace_len = int(trace_len)
        traces.append(
            build_trial_trace(
                selected_indices=selected_index_matrix[row_idx, :trace_len].tolist(),
                observed_values=observed_matrix[row_idx, :trace_len].tolist(),
                phases=(
                    None
                    if phase_matrix is None
                    else phase_matrix[row_idx, :trace_len].tolist()
                ),
            )
        )

    return traces, trial_numbers
