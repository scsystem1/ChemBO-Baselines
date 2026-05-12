from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.tabular_benchmarks import dataframe_to_one_hot


def pad_results_matrix(results: np.ndarray, target_width: int) -> np.ndarray:
    if results.shape[1] >= target_width:
        return results
    padded = np.full((results.shape[0], target_width), np.nan, dtype=float)
    padded[:, : results.shape[1]] = results
    return padded


def load_existing_results(results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not results_path.exists():
        return None
    with np.load(results_path) as payload:
        results = np.asarray(payload["results"], dtype=float)
        if "trace_lengths" in payload.files:
            trace_lengths = np.asarray(payload["trace_lengths"], dtype=int)
        else:
            trace_lengths = np.sum(~np.isnan(results), axis=1, dtype=int)
        if "trial_numbers" in payload.files:
            trial_numbers = np.asarray(payload["trial_numbers"], dtype=int)
        else:
            trial_numbers = np.arange(1, results.shape[0] + 1, dtype=int)
    return results, trace_lengths, trial_numbers


def build_scaled_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    feature_df = dataframe_to_one_hot(df, feature_columns).astype(float)
    min_values = feature_df.min(axis=0)
    ranges = feature_df.max(axis=0) - min_values
    safe_ranges = ranges.where(ranges > 0, 1.0)
    return (feature_df - min_values) / safe_ranges


def choose_initial_indices(n_rows: int, init_size: int, seed: int) -> np.ndarray:
    if init_size <= 0:
        raise ValueError("init_size must be positive.")
    if init_size > n_rows:
        raise ValueError(f"init_size={init_size} exceeds dataset size {n_rows}.")
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(n_rows, size=init_size, replace=False), dtype=int)


def nearest_candidate_index(candidate: np.ndarray, pool: np.ndarray) -> int:
    if pool.ndim != 2:
        raise ValueError("pool must have shape (n_candidates, n_features).")
    if pool.shape[0] == 0:
        raise ValueError("pool must contain at least one candidate.")
    candidate_row = np.asarray(candidate, dtype=float).reshape(1, -1)
    distances = np.sum((pool - candidate_row) ** 2, axis=1)
    return int(np.argmin(distances))
