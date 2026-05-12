from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "baseline"))

from common.experiment_tracking import (
    TrialTrace,
    build_trial_trace,
    combine_best_results,
    export_trajectory_artifacts,
    summarize_best_results,
)
from common.pool_baseline_support import (
    build_scaled_feature_frame,
    choose_initial_indices,
    load_existing_results,
    nearest_candidate_index,
)
from common.progress import progress_bar, progress_log
from common.tabular_benchmarks import load_benchmark_spec

DEFAULT_INIT_SIZE = 5
JITTER_PREFIXES = (
    "jitter =",
    "jitter is too large",
)


def log(message: str) -> None:
    progress_log(message)


class _FilteredStream:
    def __init__(self, stream, dropped_prefixes: tuple[str, ...]) -> None:
        self._stream = stream
        self._dropped_prefixes = dropped_prefixes
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not self._should_drop(line):
                self._stream.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer and not self._should_drop(self._buffer):
            self._stream.write(self._buffer)
        self._buffer = ""
        self._stream.flush()

    def _should_drop(self, line: str) -> bool:
        stripped = line.strip()
        return any(stripped.startswith(prefix) for prefix in self._dropped_prefixes)


@contextmanager
def suppress_hebo_jitter_output():
    stdout_filter = _FilteredStream(sys.stdout, JITTER_PREFIXES)
    stderr_filter = _FilteredStream(sys.stderr, JITTER_PREFIXES)
    with redirect_stdout(stdout_filter), redirect_stderr(stderr_filter):
        yield


def cap_cpu_threads(max_threads: int = 40) -> int:
    thread_cap = min(max_threads, os.cpu_count() or max_threads)
    for env_name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        current_value = os.getenv(env_name, "").strip()
        if current_value.isdigit() and int(current_value) > 0:
            thread_cap = min(thread_cap, int(current_value))
    for env_name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[env_name] = str(thread_cap)
    try:
        import torch

        torch.set_num_threads(thread_cap)
    except Exception:
        pass
    return thread_cap


def load_hebo_classes():
    script_dir = str(Path(__file__).resolve().parent)
    removed_paths: list[tuple[int, str]] = []
    for index in range(len(sys.path) - 1, -1, -1):
        if sys.path[index] in {"", script_dir}:
            removed_paths.append((index, sys.path.pop(index)))
    stale_module = sys.modules.pop("hebo", None)
    try:
        design_space_module = importlib.import_module("hebo.design_space.design_space")
        optimizer_module = importlib.import_module("hebo.optimizers.hebo")
    except ImportError as exc:
        if stale_module is not None:
            sys.modules["hebo"] = stale_module
        raise RuntimeError(
            "HEBO is not importable in the active environment. "
            "This runner expects the 'hebo' package to be installed."
        ) from exc
    finally:
        for index, path in sorted(removed_paths):
            sys.path.insert(index, path)
    return design_space_module.DesignSpace, optimizer_module.HEBO


def build_design_space(feature_df: pd.DataFrame):
    DesignSpace, _ = load_hebo_classes()
    space_config = [
        {"name": column_name, "type": "num", "lb": 0.0, "ub": 1.0}
        for column_name in feature_df.columns
    ]
    return DesignSpace().parse(space_config)


def create_hebo_optimizer(space, rand_sample: int):
    _, HEBO = load_hebo_classes()
    for kwargs in (
        {"rand_sample": rand_sample},
        {},
    ):
        try:
            return HEBO(space, **kwargs)
        except TypeError:
            continue
    return HEBO(space)


def choose_next_pool_index(optimizer, remaining_x: pd.DataFrame) -> int:
    if remaining_x.empty:
        raise ValueError("remaining_x must contain at least one candidate.")
    with suppress_hebo_jitter_output():
        suggestions = optimizer.suggest(1)
    if not isinstance(suggestions, pd.DataFrame) or suggestions.empty:
        return 0
    suggestion = suggestions.iloc[0]
    candidate = suggestion.reindex(remaining_x.columns).to_numpy(dtype=float, copy=True)
    if not np.isfinite(candidate).all():
        return 0
    return nearest_candidate_index(candidate=candidate, pool=remaining_x.to_numpy(dtype=float))


def observe_point(optimizer, point_df: pd.DataFrame, objective_value: float) -> None:
    # HEBO minimizes by default, so we negate the tabular objective that we want to maximize.
    with suppress_hebo_jitter_output():
        optimizer.observe(
            point_df.reset_index(drop=True),
            np.asarray([[-objective_value]], dtype=float),
        )


def run_trial(
    x_all: pd.DataFrame,
    y_all: np.ndarray,
    *,
    seed: int,
    total_budget: int,
    init_size: int,
    rand_sample: int,
    checkpoint_callback=None,
) -> TrialTrace:
    np.random.seed(seed)
    selected_indices = choose_initial_indices(len(y_all), init_size, seed).tolist()
    observed_values = [float(y_all[idx]) for idx in selected_indices]
    phases = ["init"] * len(selected_indices)

    design_space = build_design_space(x_all)
    optimizer = create_hebo_optimizer(design_space, rand_sample=rand_sample)
    for idx in selected_indices:
        observe_point(optimizer, x_all.iloc[[idx]], float(y_all[idx]))

    if checkpoint_callback is not None:
        checkpoint_callback(
            build_trial_trace(
                selected_indices=selected_indices,
                observed_values=observed_values,
                phases=phases,
            )
        )

    with progress_bar(total=total_budget, desc=f"HEBO seed={seed}", unit="eval") as progress:
        progress.update(len(selected_indices))
        progress.set_postfix_str(f"best={max(observed_values):.4f}")
        while len(selected_indices) < total_budget:
            selected_set = set(selected_indices)
            remaining_indices = [idx for idx in range(len(y_all)) if idx not in selected_set]
            if not remaining_indices:
                break
            remaining_x = x_all.iloc[remaining_indices].reset_index(drop=True)
            next_local_idx = choose_next_pool_index(optimizer, remaining_x)
            chosen_idx = int(remaining_indices[next_local_idx])
            chosen_value = float(y_all[chosen_idx])
            selected_indices.append(chosen_idx)
            observed_values.append(chosen_value)
            phases.append("bo")
            observe_point(optimizer, x_all.iloc[[chosen_idx]], chosen_value)
            progress.update(1)
            progress.set_postfix_str(f"best={max(observed_values):.4f}")
            log(
                f"[HEBO] Trial seed={seed} selected idx={chosen_idx} "
                f"observed={chosen_value:.4f} new_best={max(observed_values):.4f}"
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    build_trial_trace(
                        selected_indices=selected_indices,
                        observed_values=observed_values,
                        phases=phases,
                    )
                )

    return build_trial_trace(
        selected_indices=selected_indices,
        observed_values=observed_values,
        phases=phases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HEBO on a tabular chemistry dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial-start-index", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--init-size", type=int, default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--exclude-columns", default=None)
    parser.add_argument("--text-columns", default=None)
    parser.add_argument("--append-to-existing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rand-sample", type=int, default=None)
    args = parser.parse_args()

    thread_cap = cap_cpu_threads()
    resolved_init_size = args.init_size or DEFAULT_INIT_SIZE
    resolved_rand_sample = args.rand_sample or resolved_init_size
    if args.total_budget <= 0:
        raise ValueError("--total-budget must be positive.")
    if args.trial_start_index <= 0:
        raise ValueError("--trial-start-index must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative.")
    if resolved_init_size <= 0:
        raise ValueError("--init-size must be positive.")
    if resolved_rand_sample <= 0:
        raise ValueError("--rand-sample must be positive.")
    if resolved_init_size >= args.total_budget:
        raise ValueError("--init-size must be smaller than --total-budget.")

    spec, df = load_benchmark_spec(
        ROOT,
        args.dataset,
        data_path=args.data_path,
        target_column=args.target_column,
        feature_columns=args.feature_columns,
        exclude_columns=args.exclude_columns,
        text_columns=args.text_columns,
    )
    if args.total_budget > len(df):
        raise ValueError(f"--total-budget={args.total_budget} exceeds dataset size {len(df)}.")

    x_all = build_scaled_feature_frame(df, spec.feature_columns)
    y_all = df[spec.target_column].to_numpy(dtype=float)

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "hebo" / args.dataset)
    results_path = output_dir / f"{args.dataset}_hebo_results.npz"
    requested_trial_numbers = np.arange(
        args.trial_start_index,
        args.trial_start_index + args.trials,
        dtype=int,
    )
    existing_payload = load_existing_results(results_path) if args.append_to_existing else None
    if existing_payload is None:
        existing_results = None
        existing_trace_lengths = None
        existing_trial_numbers = np.empty(0, dtype=int)
    else:
        existing_results, existing_trace_lengths, existing_trial_numbers = existing_payload
        overlap = np.intersect1d(existing_trial_numbers, requested_trial_numbers)
        if overlap.size:
            overlap_text = ", ".join(f"trial_{trial_number:02d}" for trial_number in overlap.tolist())
            raise ValueError(f"Refusing to overwrite existing HEBO trial(s): {overlap_text}")

    log(
        f"[HEBO][{args.dataset.upper()}] Loaded dataset with {len(df)} rows, "
        f"budget={args.total_budget}, init_size={resolved_init_size}, trials={args.trials}, "
        f"rand_sample={resolved_rand_sample}, cpu_threads={thread_cap}"
    )
    log(f"[HEBO][{args.dataset.upper()}] Writing outputs to {output_dir}")

    def persist_snapshot(current_traces: list[TrialTrace]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        current_trial_numbers = requested_trial_numbers[: len(current_traces)]
        results, trace_lengths, trial_numbers = combine_best_results(
            current_traces,
            current_trial_numbers,
            existing_results=existing_results,
            existing_trace_lengths=existing_trace_lengths,
            existing_trial_numbers=existing_trial_numbers,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            results_path,
            results=results,
            trace_lengths=trace_lengths,
            trial_numbers=trial_numbers,
        )
        export_trajectory_artifacts(
            output_dir=output_dir,
            stem=f"{args.dataset}_hebo",
            traces=current_traces,
            trial_numbers=current_trial_numbers,
            dataset_df=df,
            target_column=spec.target_column,
        )
        initial_values, final_values = summarize_best_results(results, trace_lengths)
        summary = {
            "dataset": args.dataset,
            "trials": int(results.shape[0]),
            "trial_numbers": trial_numbers.tolist(),
            "total_budget": int(args.total_budget),
            "init_size": int(resolved_init_size),
            "rand_sample": int(resolved_rand_sample),
            "initial_mean": float(np.mean(initial_values)),
            "final_mean": float(np.mean(final_values)),
            "initial_values": initial_values.tolist(),
            "final_values": final_values.tolist(),
        }
        (output_dir / f"{args.dataset}_hebo_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return results, trace_lengths, trial_numbers, summary

    traces: list[TrialTrace] = []
    for offset in range(args.trials):
        seed = args.seed_start + offset
        trace = run_trial(
            x_all=x_all,
            y_all=y_all,
            seed=seed,
            total_budget=args.total_budget,
            init_size=resolved_init_size,
            rand_sample=resolved_rand_sample,
            checkpoint_callback=lambda partial_trace, completed_traces=traces: persist_snapshot(
                completed_traces + [partial_trace]
            ),
        )
        traces.append(trace)
        persist_snapshot(traces)

    results, trace_lengths, trial_numbers, summary = persist_snapshot(traces)
    log(
        f"[HEBO][{args.dataset.upper()}] Completed {args.trials} trial(s). "
        f"Mean best objective improved from {summary['initial_mean']:.4f} to {summary['final_mean']:.4f}."
    )


if __name__ == "__main__":
    main()
