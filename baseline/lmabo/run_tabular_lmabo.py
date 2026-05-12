from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "baseline"))
sys.path.append(str(Path(__file__).resolve().parent))

import baselines.bo_helpers as bo_helpers
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
from constants import ACQ_TYPE_MAPPING
from utils import get_shortest_distance_from_last_point

DEFAULT_INIT_SIZE = 5
DEFAULT_ACQ = "UCB"


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


def build_initial_prompt() -> str:
    choices = ", ".join(ACQ_TYPE_MAPPING.keys())
    return (
        "You are choosing a Bayesian optimization acquisition function for maximizing "
        "a chemistry objective over a fixed tabular candidate pool.\n"
        f"Available acquisition functions: {choices}.\n"
        "Respond in the format '<acquisition function>: brief reason'. "
        "For example: 'qJES: best choice because ...'."
    )


def build_iteration_prompt(
    train_x: torch.Tensor,
    observed_values: list[float],
    gp,
    remaining_iterations: int,
) -> str:
    lengthscales = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
    outputscale = float(np.asarray(gp.covar_module.outputscale.detach().cpu().numpy()).reshape(-1)[0])
    shortest_distance = (
        get_shortest_distance_from_last_point(train_x, build_unit_bounds(train_x.shape[1]))
        if train_x.shape[0] >= 2
        else 1.0
    )
    observed_array = np.asarray(observed_values, dtype=float)
    return (
        "Current optimization state:\n"
        f"- N: {len(observed_values)}\n"
        f"- Remaining iterations: {remaining_iterations}\n"
        f"- D: {train_x.shape[1]}\n"
        f"- Objective range: [{observed_array.min():.4f}, {observed_array.max():.4f}]\n"
        f"- Current best objective: {observed_array.max():.4f}\n"
        f"- Shortest normalized distance from the last point: {shortest_distance:.4f}\n"
        f"- GP lengthscales: min={lengthscales.min():.4f}, max={lengthscales.max():.4f}, "
        f"mean={lengthscales.mean():.4f}, std={lengthscales.std():.4f}\n"
        f"- GP outputscale: {outputscale:.4f}\n"
        "Choose the next acquisition function."
    )


def build_unit_bounds(dim: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros(dim, dtype=bo_helpers.dtype, device=bo_helpers.device),
            torch.ones(dim, dtype=bo_helpers.dtype, device=bo_helpers.device),
        ]
    )


def resolve_acq_type(raw_acq_type: str, default_acq: str) -> str:
    if raw_acq_type in ACQ_TYPE_MAPPING:
        return raw_acq_type
    return default_acq


def choose_seeded_random_pool_index(
    remaining_indices: list[int],
    *,
    seed: int,
    evaluation_index: int,
) -> int:
    if not remaining_indices:
        raise ValueError("remaining_indices must contain at least one candidate.")
    rng = np.random.default_rng(seed * 1_000_003 + evaluation_index)
    return int(rng.integers(len(remaining_indices)))


def choose_next_pool_index(
    acq_type: str,
    gp,
    train_y_neg: torch.Tensor,
    heldout_x: torch.Tensor,
) -> int:
    bounds = build_unit_bounds(heldout_x.shape[1])
    acq_function = bo_helpers._prepare_acquisition_function(
        acq_type,
        bounds,
        train_y_neg.min(),
        gp,
    )
    candidate = bo_helpers._optimize_acqf(acq_type, acq_function, bounds)
    candidate_np = candidate.detach().cpu().numpy().reshape(-1)
    heldout_np = heldout_x.detach().cpu().numpy()
    return nearest_candidate_index(candidate_np, heldout_np)


def build_conversation(args):
    if args.llm_mode == "none":
        return None
    from llm_helper import ConversationHolder

    last_error = None
    for attempt_idx in range(args.conversation_init_retries):
        try:
            return ConversationHolder(
                llm=args.llm_mode,
                first_prompt=build_initial_prompt(),
                full_choice_list=list(ACQ_TYPE_MAPPING.keys()),
                server_node=args.server_node,
                default_choice=args.default_acq,
                ops_model_name=args.ops_model_name,
                api_type=args.api_type,
            )
        except Exception as exc:
            last_error = exc
            log(
                f"[LMABO] Conversation init failed "
                f"(attempt {attempt_idx + 1}/{args.conversation_init_retries}): {exc}"
            )
            if attempt_idx + 1 < args.conversation_init_retries:
                time.sleep(args.conversation_init_delay_seconds)

    if args.disable_llm_on_init_failure:
        log(
            f"[LMABO] Conversation init failed after {args.conversation_init_retries} attempt(s); "
            f"falling back to llm_mode=none for this trial. Last error: {last_error}"
        )
        return None

    raise RuntimeError(
        "Failed to initialize LMABO conversation after "
        f"{args.conversation_init_retries} attempt(s). Last error: {last_error}"
    ) from last_error


def rebuild_trace_from_metadata(
    metadata_row: dict[str, object],
    *,
    y_all: np.ndarray,
    init_size: int,
) -> TrialTrace:
    seed = int(metadata_row["seed"])
    selected_indices = choose_initial_indices(len(y_all), init_size, seed).tolist()
    observed_values = [float(y_all[idx]) for idx in selected_indices]
    phases = ["init"] * len(selected_indices)

    for record in metadata_row.get("iteration_records", []):
        chosen_idx = int(record["selected_dataset_index"])
        selected_indices.append(chosen_idx)
        observed_values.append(float(y_all[chosen_idx]))
        phases.append("bo")

    return build_trial_trace(
        selected_indices=selected_indices,
        observed_values=observed_values,
        phases=phases,
    )


def run_trial(
    x_all: torch.Tensor,
    y_all: np.ndarray,
    *,
    seed: int,
    total_budget: int,
    init_size: int,
    default_acq: str,
    llm_mode: str,
    conversation,
    trial_number: int,
    checkpoint_callback=None,
) -> tuple[TrialTrace, dict[str, object]]:
    selected_indices = choose_initial_indices(len(y_all), init_size, seed).tolist()
    observed_values = [float(y_all[idx]) for idx in selected_indices]
    phases = ["init"] * len(selected_indices)
    acq_types: list[str] = []
    iteration_records: list[dict[str, object]] = []
    last_successful_gp = None

    def build_metadata() -> dict[str, object]:
        return {
            "seed": seed,
            "trial_number": trial_number,
            "llm_mode": llm_mode,
            "acq_types": list(acq_types),
            "messages": [] if conversation is None else list(conversation.messages),
            "llm_suggestion_records": (
                []
                if conversation is None
                else [dict(record) for record in conversation.suggestion_records]
            ),
            "iteration_records": [dict(record) for record in iteration_records],
        }

    if checkpoint_callback is not None:
        checkpoint_callback(
            build_trial_trace(
                selected_indices=selected_indices,
                observed_values=observed_values,
                phases=phases,
            ),
            build_metadata(),
        )

    with progress_bar(total=total_budget, desc=f"LMABO seed={seed}", unit="eval") as progress:
        progress.update(len(selected_indices))
        progress.set_postfix_str(f"best={max(observed_values):.4f}")
        while len(selected_indices) < total_budget:
            selected_set = set(selected_indices)
            remaining_indices = [idx for idx in range(len(y_all)) if idx not in selected_set]
            if not remaining_indices:
                break

            train_x = x_all[selected_indices]
            train_y_neg = torch.tensor(
                [-value for value in observed_values],
                dtype=bo_helpers.dtype,
                device=bo_helpers.device,
            )
            heldout_x = x_all[remaining_indices]
            eval_index = len(selected_indices) + 1
            gp = None
            gp_fit_diagnostics: dict[str, object] | None = None
            gp_fit_failed = False
            gp_fit_error = None
            reused_previous_gp = False
            prompt = None
            llm_suggestion_record = None
            requested_acq = default_acq
            applied_acq = default_acq
            fallback_used = False
            fallback_reason = None
            used_random_fallback = False

            try:
                gp, gp_fit_diagnostics = bo_helpers.fit_gp(
                    train_x,
                    train_y_neg,
                    return_diagnostics=True,
                )
                last_successful_gp = gp
            except Exception as exc:
                gp_fit_failed = True
                gp_fit_error = str(exc)
                gp_fit_diagnostics = getattr(exc, "fit_diagnostics", None)
                if last_successful_gp is not None:
                    gp = last_successful_gp
                    reused_previous_gp = True
                    fallback_used = True
                    fallback_reason = (
                        f"Current GP fit failed; reused previous successful GP and forced "
                        f"default acquisition {default_acq}."
                    )
                    log(f"[LMABO] GP fit failed at eval={eval_index}; reusing previous GP. Error: {exc}")
                else:
                    fallback_used = True
                    fallback_reason = (
                        "Current GP fit failed and no previous successful GP was available; "
                        "falling back to a seeded random unseen candidate."
                    )
                    log(f"[LMABO] GP fit failed at eval={eval_index}; no reusable GP. Error: {exc}")

            if gp is not None and not reused_previous_gp:
                if llm_mode == "none":
                    requested_acq = default_acq
                else:
                    prompt = build_iteration_prompt(
                        train_x=train_x,
                        observed_values=observed_values,
                        gp=gp,
                        remaining_iterations=total_budget - len(selected_indices),
                    )
                    requested_acq = resolve_acq_type(
                        conversation.suggest_acq_type(prompt),
                        default_acq,
                    )
                    llm_suggestion_record = (
                        None
                        if conversation.last_suggestion_record is None
                        else dict(conversation.last_suggestion_record)
                    )
            elif gp is not None:
                requested_acq = default_acq

            next_local_idx = None
            if gp is not None:
                applied_acq = requested_acq
                try:
                    next_local_idx = choose_next_pool_index(applied_acq, gp, train_y_neg, heldout_x)
                except Exception as exc:
                    if applied_acq != default_acq:
                        fallback_used = True
                        fallback_reason = (
                            f"Requested acquisition {applied_acq} failed; retried with default "
                            f"acquisition {default_acq}. Error: {exc}"
                        )
                        log(
                            f"[LMABO] Acquisition {applied_acq} failed at eval={eval_index}; "
                            f"retrying with {default_acq}. Error: {exc}"
                        )
                        applied_acq = default_acq
                        try:
                            next_local_idx = choose_next_pool_index(
                                applied_acq,
                                gp,
                                train_y_neg,
                                heldout_x,
                            )
                        except Exception as default_exc:
                            fallback_used = True
                            used_random_fallback = True
                            fallback_reason = (
                                f"Default acquisition {default_acq} also failed after requested "
                                f"acquisition {requested_acq} failed. Final error: {default_exc}"
                            )
                            log(
                                f"[LMABO] Default acquisition {default_acq} also failed at "
                                f"eval={eval_index}; using seeded random fallback. Error: {default_exc}"
                            )
                            next_local_idx = choose_seeded_random_pool_index(
                                remaining_indices,
                                seed=seed,
                                evaluation_index=eval_index,
                            )
                    else:
                        fallback_used = True
                        used_random_fallback = True
                        fallback_reason = (
                            f"Default acquisition {default_acq} failed; using seeded random "
                            f"fallback. Error: {exc}"
                        )
                        log(
                            f"[LMABO] Default acquisition {default_acq} failed at eval={eval_index}; "
                            f"using seeded random fallback. Error: {exc}"
                        )
                        next_local_idx = choose_seeded_random_pool_index(
                            remaining_indices,
                            seed=seed,
                            evaluation_index=eval_index,
                        )
            else:
                applied_acq = default_acq
                used_random_fallback = True
                next_local_idx = choose_seeded_random_pool_index(
                    remaining_indices,
                    seed=seed,
                    evaluation_index=eval_index,
                )

            acq_types.append(applied_acq)
            chosen_idx = int(remaining_indices[next_local_idx])
            selected_indices.append(chosen_idx)
            observed_values.append(float(y_all[chosen_idx]))
            phases.append("bo")
            current_best = max(observed_values)
            iteration_records.append(
                {
                    "evaluation_index": int(eval_index),
                    "requested_acq": requested_acq,
                    "applied_acq": applied_acq,
                    "llm_suggestion": llm_suggestion_record,
                    "gp_fit_failed": bool(gp_fit_failed),
                    "gp_fit_error": gp_fit_error,
                    "gp_fit_diagnostics": gp_fit_diagnostics,
                    "reused_previous_gp": bool(reused_previous_gp),
                    "fallback_used": bool(fallback_used),
                    "fallback_reason": fallback_reason,
                    "used_random_fallback": bool(used_random_fallback),
                    "selected_dataset_index": int(chosen_idx),
                    "observed_objective": float(y_all[chosen_idx]),
                    "best_objective_after": float(current_best),
                    "remaining_candidates_before_selection": int(len(remaining_indices)),
                }
            )
            progress.update(1)
            progress.set_postfix_str(f"best={current_best:.4f}, acq={applied_acq}")
            log(
                f"[LMABO] Trial seed={seed} acq={applied_acq} selected idx={chosen_idx} "
                f"observed={float(y_all[chosen_idx]):.4f} new_best={current_best:.4f}"
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    build_trial_trace(
                        selected_indices=selected_indices,
                        observed_values=observed_values,
                        phases=phases,
                    ),
                    build_metadata(),
                )

    trace = build_trial_trace(
        selected_indices=selected_indices,
        observed_values=observed_values,
        phases=phases,
    )
    metadata = {
        **build_metadata(),
    }
    return trace, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LMABO on a tabular chemistry dataset.")
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
    parser.add_argument("--llm-mode", choices=["none", "api", "ops"], default="none")
    parser.add_argument("--default-acq", default=DEFAULT_ACQ)
    parser.add_argument("--api-type", choices=["gemini", "gpt"], default="gemini")
    parser.add_argument("--server-node", default="localhost")
    parser.add_argument("--ops-model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--conversation-init-retries", type=int, default=3)
    parser.add_argument("--conversation-init-delay-seconds", type=float, default=5.0)
    parser.add_argument("--disable-llm-on-init-failure", action="store_true", default=True)
    parser.add_argument("--no-disable-llm-on-init-failure", dest="disable_llm_on_init_failure", action="store_false")
    parser.add_argument("--replace-incomplete-last-trial", action="store_true")
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
    if args.conversation_init_retries <= 0:
        raise ValueError("--conversation-init-retries must be positive.")
    if args.conversation_init_delay_seconds < 0:
        raise ValueError("--conversation-init-delay-seconds must be non-negative.")
    args.default_acq = resolve_acq_type(args.default_acq, DEFAULT_ACQ)

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
    x_all = torch.tensor(
        x_df.to_numpy(dtype=float),
        dtype=bo_helpers.dtype,
        device=bo_helpers.device,
    )
    y_all = df[spec.target_column].to_numpy(dtype=float)

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "lmabo" / args.dataset)
    results_path = output_dir / f"{args.dataset}_lmabo_results.npz"
    metadata_path = output_dir / f"{args.dataset}_lmabo_trial_metadata.json"
    if args.append_to_existing and metadata_path.exists():
        existing_metadata_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        existing_metadata_rows = []
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
        if args.replace_incomplete_last_trial and existing_trial_numbers.size:
            incomplete_mask = np.asarray(existing_trace_lengths, dtype=int) < int(args.total_budget)
            if np.any(incomplete_mask):
                incomplete_trial_numbers = existing_trial_numbers[incomplete_mask]
                keep_mask = ~incomplete_mask
                existing_results = existing_results[keep_mask]
                existing_trace_lengths = existing_trace_lengths[keep_mask]
                existing_trial_numbers = existing_trial_numbers[keep_mask]
                incomplete_trial_number_set = set(incomplete_trial_numbers.tolist())
                existing_metadata_rows = [
                    row
                    for row in existing_metadata_rows
                    if int(row.get("trial_number", -1)) not in incomplete_trial_number_set
                ]
                log(
                    f"[LMABO][{args.dataset.upper()}] Replacing incomplete existing trial(s): "
                    + ", ".join(f"trial_{int(trial_number):02d}" for trial_number in incomplete_trial_numbers)
                )
        overlap = np.intersect1d(existing_trial_numbers, requested_trial_numbers)
        if overlap.size:
            overlap_text = ", ".join(f"trial_{trial_number:02d}" for trial_number in overlap.tolist())
            raise ValueError(f"Refusing to overwrite existing LMABO trial(s): {overlap_text}")

    existing_trace_trial_numbers = existing_trial_numbers.tolist()
    if existing_trace_trial_numbers:
        metadata_by_trial = {
            int(row["trial_number"]): row
            for row in existing_metadata_rows
        }
        existing_export_traces = [
            rebuild_trace_from_metadata(
                metadata_by_trial[trial_number],
                y_all=y_all,
                init_size=resolved_init_size,
            )
            for trial_number in existing_trace_trial_numbers
            if trial_number in metadata_by_trial
        ]
    else:
        existing_export_traces = []

    log(
        f"[LMABO][{args.dataset.upper()}] Loaded dataset with {len(df)} rows, "
        f"budget={args.total_budget}, init_size={resolved_init_size}, trials={args.trials}, "
        f"llm_mode={args.llm_mode}, cpu_threads={thread_cap}"
    )
    log(f"[LMABO][{args.dataset.upper()}] Writing outputs to {output_dir}")

    def persist_snapshot(
        current_traces: list[TrialTrace],
        current_metadata_rows: list[dict[str, object]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
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
            stem=f"{args.dataset}_lmabo",
            traces=all_export_traces,
            trial_numbers=all_export_trial_numbers,
            dataset_df=df,
            target_column=spec.target_column,
        )
        full_metadata_rows = existing_metadata_rows + current_metadata_rows
        metadata_path.write_text(
            json.dumps(full_metadata_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        initial_values, final_values = summarize_best_results(results, trace_lengths)
        summary = {
            "dataset": args.dataset,
            "trials": int(results.shape[0]),
            "trial_numbers": trial_numbers.tolist(),
            "total_budget": int(args.total_budget),
            "init_size": int(resolved_init_size),
            "llm_mode": args.llm_mode,
            "default_acq": args.default_acq,
            "initial_mean": float(np.mean(initial_values)),
            "final_mean": float(np.mean(final_values)),
            "initial_values": initial_values.tolist(),
            "final_values": final_values.tolist(),
        }
        (output_dir / f"{args.dataset}_lmabo_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return results, trace_lengths, trial_numbers, summary

    traces: list[TrialTrace] = []
    metadata_rows: list[dict[str, object]] = []
    for offset in range(args.trials):
        seed = args.seed_start + offset
        trial_number = int(requested_trial_numbers[offset])
        conversation = build_conversation(args)
        effective_llm_mode = args.llm_mode if conversation is not None else "none"
        trace, metadata = run_trial(
            x_all=x_all,
            y_all=y_all,
            seed=seed,
            total_budget=args.total_budget,
            init_size=resolved_init_size,
            default_acq=args.default_acq,
            llm_mode=effective_llm_mode,
            conversation=conversation,
            trial_number=trial_number,
            checkpoint_callback=lambda partial_trace, partial_metadata, completed_traces=traces, completed_metadata=metadata_rows: persist_snapshot(
                completed_traces + [partial_trace],
                completed_metadata + [partial_metadata],
            ),
        )
        traces.append(trace)
        metadata_rows.append(metadata)
        persist_snapshot(traces, metadata_rows)

    results, trace_lengths, trial_numbers, summary = persist_snapshot(traces, metadata_rows)
    log(
        f"[LMABO][{args.dataset.upper()}] Completed {args.trials} trial(s). "
        f"Mean best objective improved from {summary['initial_mean']:.4f} to {summary['final_mean']:.4f}."
    )


if __name__ == "__main__":
    main()
