from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BOOST_ROOT = ROOT / "baseline" / "BOOST_AutoML"
BOOST_CODE = BOOST_ROOT / "Code"
for import_path in [BOOST_CODE, BOOST_CODE / "tests"]:
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

sys.path.append(str(ROOT / "baseline"))

from common.experiment_tracking import (
    TrialTrace,
    build_trial_trace,
    combine_best_results,
    export_trajectory_artifacts,
    load_existing_trajectory_artifacts,
    summarize_best_results,
)
from common.pool_baseline_support import (
    build_scaled_feature_frame,
    choose_initial_indices,
    load_existing_results,
)
from common.progress import progress_bar, progress_log
from common.tabular_benchmarks import load_benchmark_spec
from core.BOOST import BOOST
from core.BayesianOptimization import BayesianOptimizer


DEFAULT_INIT_SIZE = 10


def log(message: str) -> None:
    progress_log(message)


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
        torch.set_num_threads(thread_cap)
    except Exception:
        pass
    return thread_cap


def objective_to_boost_minimization_values(y_all: np.ndarray) -> torch.Tensor:
    y_values = np.asarray(y_all, dtype=float)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if y_max <= y_min:
        scaled = np.zeros_like(y_values, dtype=float)
    else:
        scaled = (y_values - y_min) / (y_max - y_min)
    return torch.tensor(-scaled, dtype=torch.double)


def run_trial(
    x_all: torch.Tensor,
    y_all: np.ndarray,
    boost_y_all: torch.Tensor,
    *,
    dataset: str,
    seed: int,
    trial_number: int,
    total_budget: int,
    init_size: int,
    device: str,
    recommendation_log_dir: Path,
    checkpoint_callback=None,
) -> TrialTrace:
    selected_indices = choose_initial_indices(len(y_all), init_size, seed).tolist()
    observed_values = [float(y_all[idx]) for idx in selected_indices]
    phases = ["init"] * len(selected_indices)

    selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=x_all.device)
    train_x = x_all[selected_tensor]
    train_y = boost_y_all[selected_tensor]

    remaining_indices = [idx for idx in range(len(y_all)) if idx not in set(selected_indices)]
    filtered_candidate_x = x_all[remaining_indices]
    filtered_candidate_y = boost_y_all[remaining_indices]

    outer_bo = BayesianOptimizer(device=device)
    boost = BOOST(device=device, is_fixed_candidate_x=True)

    if checkpoint_callback is not None:
        checkpoint_callback(
            build_trial_trace(
                selected_indices=selected_indices,
                observed_values=observed_values,
                phases=phases,
            )
        )

    with progress_bar(total=total_budget, desc=f"BOOST seed={seed}", unit="eval") as progress:
        progress.update(len(selected_indices))
        progress.set_postfix_str(f"best={max(observed_values):.4f}")

        while len(selected_indices) < total_budget and remaining_indices:
            eval_index = len(selected_indices)
            kernel_type, acquisition_type = boost.get_kernel_acq(
                train_x=train_x,
                train_y=train_y,
                objective=dataset,
                iter=eval_index,
                seed=seed,
                n_init_points=init_size,
                base_dir=str(recommendation_log_dir),
            )
            next_x, next_y, next_local_idx = outer_bo.get_next_point(
                train_x=train_x,
                train_y=train_y,
                filtered_candidate_x=filtered_candidate_x,
                filtered_candidate_y=filtered_candidate_y,
                kernel_type=kernel_type,
                acquisition_type=acquisition_type,
                objective=None,
            )

            next_local_idx_int = int(next_local_idx.item() if torch.is_tensor(next_local_idx) else next_local_idx)
            chosen_idx = int(remaining_indices[next_local_idx_int])
            selected_indices.append(chosen_idx)
            observed_values.append(float(y_all[chosen_idx]))
            phases.append("bo")

            train_x = torch.cat([train_x, next_x], dim=0)
            train_y = torch.cat([train_y, next_y], dim=0)

            mask = torch.ones(filtered_candidate_x.shape[0], dtype=torch.bool, device=x_all.device)
            mask[next_local_idx_int] = False
            filtered_candidate_x = filtered_candidate_x[mask]
            filtered_candidate_y = filtered_candidate_y[mask]
            del remaining_indices[next_local_idx_int]

            current_best = max(observed_values)
            progress.update(1)
            progress.set_postfix_str(
                f"best={current_best:.4f}, kernel={kernel_type.value}, acq={acquisition_type.value}"
            )
            log(
                f"[BOOST] Trial {trial_number} seed={seed} kernel={kernel_type.value} "
                f"acq={acquisition_type.value} selected idx={chosen_idx} "
                f"observed={float(y_all[chosen_idx]):.4f} new_best={current_best:.4f}"
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
    parser = argparse.ArgumentParser(description="Run BOOST on a tabular chemistry dataset.")
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
    parser.add_argument("--replace-incomplete-last-trial", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    thread_cap = cap_cpu_threads()
    resolved_init_size = args.init_size or DEFAULT_INIT_SIZE
    if args.total_budget <= 0:
        raise ValueError("--total-budget must be positive.")
    if args.trial_start_index <= 0:
        raise ValueError("--trial-start-index must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative.")
    if resolved_init_size <= 0:
        raise ValueError("--init-size must be positive.")
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

    x_df = build_scaled_feature_frame(df, spec.feature_columns)
    x_all = torch.tensor(x_df.to_numpy(dtype=float), dtype=torch.double)
    if args.device == "cuda":
        x_all = x_all.cuda()
    y_all = df[spec.target_column].to_numpy(dtype=float)
    boost_y_all = objective_to_boost_minimization_values(y_all).to(x_all.device)

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "boost" / args.dataset)
    results_path = output_dir / f"{args.dataset}_boost_results.npz"
    trajectory_path = output_dir / f"{args.dataset}_boost_trajectory.npz"
    recommendation_log_dir = output_dir / "recommendation_logs"
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

    existing_trace_trial_numbers = existing_trial_numbers.tolist()
    existing_trajectory_payload = (
        load_existing_trajectory_artifacts(trajectory_path)
        if args.append_to_existing
        else None
    )
    if existing_trajectory_payload is None:
        existing_export_traces = []
        existing_trace_trial_numbers = []
    else:
        existing_export_traces, existing_export_trial_numbers = existing_trajectory_payload
        existing_trace_trial_numbers = np.asarray(existing_export_trial_numbers, dtype=int).tolist()

    if args.replace_incomplete_last_trial and existing_trial_numbers.size:
        incomplete_mask = np.asarray(existing_trace_lengths, dtype=int) < int(args.total_budget)
        if np.any(incomplete_mask):
            incomplete_trial_numbers = existing_trial_numbers[incomplete_mask]
            keep_mask = ~incomplete_mask
            existing_results = existing_results[keep_mask]
            existing_trace_lengths = existing_trace_lengths[keep_mask]
            existing_trial_numbers = existing_trial_numbers[keep_mask]
            incomplete_trial_number_set = set(incomplete_trial_numbers.tolist())
            existing_export_pairs = [
                (trial_number, trace)
                for trial_number, trace in zip(existing_trace_trial_numbers, existing_export_traces)
                if int(trial_number) not in incomplete_trial_number_set
            ]
            existing_trace_trial_numbers = [int(trial_number) for trial_number, _ in existing_export_pairs]
            existing_export_traces = [trace for _, trace in existing_export_pairs]
            log(
                f"[BOOST][{args.dataset.upper()}] Replacing incomplete existing trial(s): "
                + ", ".join(f"trial_{int(trial_number):02d}" for trial_number in incomplete_trial_numbers)
            )

    overlap = np.intersect1d(existing_trial_numbers, requested_trial_numbers)
    if overlap.size:
        overlap_text = ", ".join(f"trial_{trial_number:02d}" for trial_number in overlap.tolist())
        raise ValueError(f"Refusing to overwrite existing BOOST trial(s): {overlap_text}")

    log(
        f"[BOOST][{args.dataset.upper()}] Loaded dataset with {len(df)} rows, "
        f"budget={args.total_budget}, init_size={resolved_init_size}, trials={args.trials}, "
        f"features={x_all.shape[1]}, device={args.device}, cpu_threads={thread_cap}"
    )
    log(f"[BOOST][{args.dataset.upper()}] Writing outputs to {output_dir}")

    def persist_snapshot(current_traces: list[TrialTrace]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        current_trial_numbers = requested_trial_numbers[: len(current_traces)]
        all_export_traces = existing_export_traces + current_traces
        all_export_trial_numbers = np.concatenate(
            [
                np.asarray(existing_trace_trial_numbers, dtype=int),
                current_trial_numbers,
            ]
        )
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
            stem=f"{args.dataset}_boost",
            traces=all_export_traces,
            trial_numbers=all_export_trial_numbers,
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
            "feature_count": int(x_all.shape[1]),
            "target_column": spec.target_column,
            "objective_direction": "maximize",
            "boost_internal_objective": "-minmax_scaled_objective",
            "initial_mean": float(np.mean(initial_values)),
            "final_mean": float(np.mean(final_values)),
            "initial_values": initial_values.tolist(),
            "final_values": final_values.tolist(),
        }
        (output_dir / f"{args.dataset}_boost_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return results, trace_lengths, trial_numbers, summary

    traces: list[TrialTrace] = []
    for offset in range(args.trials):
        trial_number = int(requested_trial_numbers[offset])
        seed = args.seed_start + offset
        trace = run_trial(
            x_all=x_all,
            y_all=y_all,
            boost_y_all=boost_y_all,
            dataset=args.dataset,
            seed=seed,
            trial_number=trial_number,
            total_budget=args.total_budget,
            init_size=resolved_init_size,
            device=args.device,
            recommendation_log_dir=recommendation_log_dir,
            checkpoint_callback=lambda partial_trace, completed_traces=traces: persist_snapshot(
                completed_traces + [partial_trace]
            ),
        )
        traces.append(trace)
        persist_snapshot(traces)

    results, trace_lengths, trial_numbers, summary = persist_snapshot(traces)
    log(
        f"[BOOST][{args.dataset.upper()}] Completed {args.trials} trial(s). "
        f"Mean best objective improved from {summary['initial_mean']:.4f} to {summary['final_mean']:.4f}."
    )


if __name__ == "__main__":
    main()
