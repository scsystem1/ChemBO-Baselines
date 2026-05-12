from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ax import ChoiceParameter, Experiment, Objective, OptimizationConfig, ParameterType, Runner, SearchSpace

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "baseline"))
sys.path.append(str(Path(__file__).resolve().parent))

from common.experiment_tracking import (
    TrialTrace,
    build_best_results_matrix,
    build_trial_trace,
    export_trajectory_artifacts,
    summarize_best_results,
)
from common.progress import progress_bar, progress_log
from common.tabular_benchmarks import load_benchmark_spec
from src.bo.models import BOModel
from src.bo.reasoner.kimi import KimiReasoner
from src.tasks.chemistry.tabular import TabularChemistryMetric
from src.utils.metric import save_trial_data


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


def _pad_results_matrix(results: np.ndarray, target_width: int) -> np.ndarray:
    if results.shape[1] >= target_width:
        return results
    padded = np.full((results.shape[0], target_width), np.nan, dtype=float)
    padded[:, : results.shape[1]] = results
    return padded


def _load_existing_results(results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_templates(dataset_name: str) -> tuple[str, str, str]:
    if dataset_name == "dar":
        return (
            "Direct Arylation Reaction Optimization",
            "Chemistry/Reaction Condition Optimization",
            "Optimize direct arylation yield by selecting base, ligand, solvent, concentration, and temperature combinations from a tabulated experimental design space.",
        )
    if dataset_name == "ocm":
        return (
            "Oxidative Coupling of Methane Catalyst Optimization",
            "Chemistry/Heterogeneous Catalysis Optimization",
            "Optimize oxidative coupling of methane performance by selecting catalyst composition, support, and operating conditions from a tabulated experimental design space.",
        )
    if dataset_name == "suzuki":
        return (
            "Suzuki-Miyaura Cross-Coupling Optimization",
            "Chemistry/Cross-Coupling Reaction Optimization",
            "Optimize Suzuki-Miyaura cross-coupling yield by selecting reactants, ligand, base, and solvent combinations from a tabulated experimental design space.",
        )
    dataset_label = dataset_name.strip().upper()
    return (
        f"{dataset_label} Tabular Optimization",
        "Chemistry/Tabular Optimization",
        f"Optimize {dataset_label} target values by selecting combinations from a tabulated experimental design space.",
    )


def infer_parameter_type(values: list[object]) -> ParameterType:
    non_null = [value for value in values if pd.notna(value)]
    if all(isinstance(value, (int, np.integer)) for value in non_null):
        return ParameterType.INT
    if all(isinstance(value, (int, float, np.integer, np.floating)) for value in non_null):
        return ParameterType.FLOAT
    return ParameterType.STRING


def normalize_choice_value(value: object, parameter_type: ParameterType) -> object:
    if pd.isna(value):
        if parameter_type == ParameterType.STRING:
            return "nan"
        return value
    if parameter_type == ParameterType.INT:
        return int(round(float(value)))
    if parameter_type == ParameterType.FLOAT:
        return float(value)
    return str(value)


def build_search_space_and_config(
    dataset_name: str,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[SearchSpace, Path, dict[str, dict[str, object]]]:
    exp_name, app_context, description = dataset_templates(dataset_name)
    parameter_definitions = []
    parameters = []
    parameter_specs: dict[str, dict[str, object]] = {}

    for column in feature_columns:
        raw_values = df[column].tolist()
        unique_values = list(dict.fromkeys(raw_values))
        parameter_type = infer_parameter_type(unique_values)
        normalized_values = [normalize_choice_value(value, parameter_type) for value in unique_values]
        deduped_values = list(dict.fromkeys(normalized_values))
        bounds = sorted(deduped_values) if parameter_type != ParameterType.STRING else sorted(deduped_values, key=str)
        parameters.append(
            ChoiceParameter(
                name=column,
                parameter_type=parameter_type,
                values=bounds,
            )
        )
        parameter_specs[column] = {"type": parameter_type, "values": bounds}
        parameter_definitions.append(
            {
                "display_name": column,
                "description": f"Allowed values for parameter {column}.",
                "data_type": parameter_type.name.lower(),
                "bounds": bounds,
            }
        )

    config_payload = {
        "name": exp_name,
        "application_context": app_context,
        "description": description,
        "constraint": "All parameter values must come exactly from the predefined design space.",
        "parameter_definitions": parameter_definitions,
        "target": {
            "name": target_column,
            "description": f"Objective column {target_column} from the tabulated benchmark.",
            "direction": "maximize",
        },
    }
    config_path = output_dir / f"{dataset_name}_reasoning_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return SearchSpace(parameters=parameters), config_path, parameter_specs


class StaticRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}


def arm_name_key(name: str) -> list[int]:
    return [int(piece) for piece in name.split("_")]


def update_trace_from_trial(trace: list[float], trial) -> list[float]:
    trial_df = trial.fetch_data().df.sort_values("arm_name", key=lambda s: s.map(arm_name_key))
    current_best = trace[-1] if trace else float("-inf")
    for mean in trial_df["mean"].tolist():
        current_best = max(current_best, float(mean))
        trace.append(current_best)
    return trace


def candidate_to_dataset_index(
    candidate: dict[str, object],
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> int:
    candidate_rows = feature_df.loc[:, feature_columns].copy()
    mask = pd.Series(True, index=candidate_rows.index)
    for column in feature_columns:
        column_values = candidate_rows[column]
        value = candidate[column]
        if pd.api.types.is_numeric_dtype(column_values):
            mask &= np.isclose(column_values.astype(float), float(value))
        else:
            mask &= column_values.astype(str) == str(value)
    matches = candidate_rows.index[mask].tolist()
    if not matches:
        raise RuntimeError(f"Unable to map candidate back to dataset row: {candidate}")
    return int(matches[0])


def canonicalize_candidate(candidate: dict[str, object], parameter_specs: dict[str, dict[str, object]]) -> dict[str, object] | None:
    canonical: dict[str, object] = {}
    for name, spec in parameter_specs.items():
        if name not in candidate:
            return None
        value = candidate[name]
        allowed_values = spec["values"]
        parameter_type = spec["type"]
        try:
            if parameter_type == ParameterType.INT:
                normalized = int(round(float(value)))
            elif parameter_type == ParameterType.FLOAT:
                normalized = float(value)
            else:
                normalized = str(value)
        except Exception:
            return None
        if normalized not in allowed_values:
            return None
        canonical[name] = normalized
    return canonical


def project_candidate_to_design_row(
    candidate: dict[str, object],
    feature_df: pd.DataFrame,
    parameter_specs: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    raw: dict[str, object] = {}
    for name, spec in parameter_specs.items():
        if name not in candidate:
            return None
        value = candidate[name]
        try:
            if spec["type"] == ParameterType.INT:
                raw[name] = int(round(float(value)))
            elif spec["type"] == ParameterType.FLOAT:
                raw[name] = float(value)
            else:
                raw[name] = str(value)
        except Exception:
            return None

    exact = canonicalize_candidate(raw, parameter_specs)
    if exact is not None:
        row_key = tuple(exact[name] for name in feature_df.columns)
        feature_keys = {
            tuple(normalize_choice_value(row[name], parameter_specs[name]["type"]) for name in feature_df.columns)
            for _, row in feature_df.iterrows()
        }
        if row_key in feature_keys:
            return exact

    candidate_rows = feature_df.copy()
    categorical_cols = [
        name
        for name, spec in parameter_specs.items()
        if spec["type"] == ParameterType.STRING
    ]
    for name in categorical_cols:
        matches = candidate_rows[name].astype(str) == str(raw[name])
        if matches.any():
            candidate_rows = candidate_rows.loc[matches]

    if candidate_rows.empty:
        candidate_rows = feature_df

    numeric_cols = [
        name
        for name, spec in parameter_specs.items()
        if spec["type"] in (ParameterType.INT, ParameterType.FLOAT)
    ]
    ranges: dict[str, float] = {}
    for name in numeric_cols:
        series = feature_df[name].astype(float)
        ranges[name] = max(float(series.max()) - float(series.min()), 1.0)

    best_score = None
    best_row = None
    for _, row in candidate_rows.iterrows():
        score = 0.0
        for name in categorical_cols:
            if str(row[name]) != str(raw[name]):
                score += 1e6
        for name in numeric_cols:
            score += ((float(row[name]) - float(raw[name])) / ranges[name]) ** 2
        if best_score is None or score < best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None

    projected = {
        name: normalize_choice_value(best_row[name], parameter_specs[name]["type"])
        for name in feature_df.columns
    }
    return projected


def candidate_key(candidate: dict[str, object], feature_columns: list[str]) -> tuple[object, ...]:
    return tuple(candidate[name] for name in feature_columns)


def evaluated_keys_from_experiment(experiment: Experiment, feature_columns: list[str]) -> set[tuple[object, ...]]:
    seen: set[tuple[object, ...]] = set()
    for trial in experiment.trials.values():
        for arm in trial.arms:
            if all(name in arm.parameters for name in feature_columns):
                seen.add(candidate_key(arm.parameters, feature_columns))
    return seen


def fill_candidates(
    candidates: list[dict[str, object]],
    feature_df: pd.DataFrame,
    parameter_specs: dict[str, dict[str, object]],
    bo_model: BOModel,
    experiment: Experiment,
    n_needed: int,
) -> list[dict[str, object]]:
    valid: list[dict[str, object]] = []
    feature_columns = list(feature_df.columns)
    seen = evaluated_keys_from_experiment(experiment, feature_columns)
    for candidate in candidates:
        canonical = project_candidate_to_design_row(
            candidate=candidate,
            feature_df=feature_df,
            parameter_specs=parameter_specs,
        )
        if canonical is None:
            continue
        key = candidate_key(canonical, feature_columns)
        if key not in seen:
            valid.append(canonical)
            seen.add(key)

    if len(valid) < n_needed:
        generator_run = bo_model.gen(n=n_needed)
        for arm in generator_run.arms:
            canonical = project_candidate_to_design_row(
                candidate=arm.parameters,
                feature_df=feature_df,
                parameter_specs=parameter_specs,
            )
            if canonical is None:
                continue
            key = candidate_key(canonical, feature_columns)
            if key not in seen:
                valid.append(canonical)
                seen.add(key)
            if len(valid) >= n_needed:
                break

    if len(valid) < n_needed:
        candidate_pool = feature_df.drop_duplicates().sample(
            frac=1.0, random_state=np.random.randint(0, 2**31 - 1)
        )
        for _, row in candidate_pool.iterrows():
            canonical = {
                name: normalize_choice_value(row[name], parameter_specs[name]["type"])
                for name in feature_columns
            }
            key = candidate_key(canonical, feature_columns)
            if key in seen:
                continue
            valid.append(canonical)
            seen.add(key)
            if len(valid) >= n_needed:
                break

    if len(valid) < n_needed:
        raise RuntimeError(f"Unable to assemble {n_needed} valid candidates; only got {len(valid)}.")
    return valid[:n_needed]


def normalize_candidates(
    candidates: list[dict[str, object]] | None,
    feature_df: pd.DataFrame,
    parameter_specs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    valid: list[dict[str, object]] = []
    seen = set()
    for candidate in candidates or []:
        canonical = project_candidate_to_design_row(
            candidate=candidate,
            feature_df=feature_df,
            parameter_specs=parameter_specs,
        )
        if canonical is None:
            continue
        key = tuple((name, canonical[name]) for name in sorted(canonical))
        if key in seen:
            continue
        valid.append(canonical)
        seen.add(key)
    return valid


def _fill_initial_candidates(
    initial_candidates: list[dict[str, object]],
    feature_df: pd.DataFrame,
    parameter_specs: dict[str, dict[str, object]],
    target_count: int,
    seed: int,
) -> list[dict[str, object]]:
    if len(initial_candidates) >= target_count:
        return initial_candidates[:target_count]
    valid = list(initial_candidates)
    seen = {
        tuple((name, candidate[name]) for name in feature_df.columns)
        for candidate in valid
    }
    candidate_pool = feature_df.drop_duplicates().sample(frac=1.0, random_state=seed)
    for _, row in candidate_pool.iterrows():
        candidate = {
            name: normalize_choice_value(row[name], parameter_specs[name]["type"])
            for name in feature_df.columns
        }
        key = tuple((name, candidate[name]) for name in feature_df.columns)
        if key in seen:
            continue
        valid.append(candidate)
        seen.add(key)
        if len(valid) >= target_count:
            break
    if len(valid) < target_count:
        raise RuntimeError(
            f"Unable to assemble {target_count} initial candidates; only got {len(valid)}."
        )
    return valid


def record_batch_observations(
    trial,
    candidates: list[dict[str, object]],
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    selected_indices: list[int],
    observed_values: list[float],
    phases: list[str],
    phase: str,
) -> None:
    trial_df = trial.fetch_data().df.sort_values("arm_name", key=lambda s: s.map(arm_name_key))
    means = [float(value) for value in trial_df["mean"].tolist()]
    if len(means) != len(candidates):
        raise RuntimeError(
            f"Expected {len(candidates)} observations for batch, got {len(means)}."
        )
    for candidate, mean in zip(candidates, means):
        selected_indices.append(
            candidate_to_dataset_index(candidate, feature_df, feature_columns)
        )
        observed_values.append(mean)
        phases.append(phase)


def run_trial(
    dataset_name: str,
    df: pd.DataFrame,
    data_path: Path,
    target_column: str,
    feature_columns: list[str],
    total_budget: int,
    init_size: int,
    reasoning_batch_size: int,
    use_llm_initial_count: bool,
    allow_batch_overshoot: bool,
    seed: int,
    output_dir: Path,
) -> TrialTrace:
    seed_everything(seed)
    feature_df = df.loc[:, feature_columns].copy()
    search_space, config_path, parameter_specs = build_search_space_and_config(
        dataset_name=dataset_name,
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        output_dir=output_dir,
    )
    reasoner = KimiReasoner(exp_config_path=str(config_path), result_dir=str(output_dir))
    experiment = Experiment(
        name=f"{dataset_name}_reasoning_bo_seed{seed}",
        search_space=search_space,
        optimization_config=OptimizationConfig(
            objective=Objective(
                metric=TabularChemistryMetric(
                    name=dataset_name,
                    data_path=data_path,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    noiseless=True,
                )
            )
        ),
        runner=StaticRunner(),
    )
    bo_model = BOModel(experiment)
    selected_indices: list[int] = []
    observed_values: list[float] = []
    phases: list[str] = []

    progress_total = total_budget if not allow_batch_overshoot else total_budget + reasoning_batch_size - 1
    with progress_bar(total=progress_total, desc=f"ReasoningBO seed={seed}", unit="eval") as progress:
        log(f"[ReasoningBO][{dataset_name.upper()}] Trial {seed + 1}: generating overview")
        reasoner.generate_overview()
        log(f"[ReasoningBO][{dataset_name.upper()}] Trial {seed + 1}: initial reasoning")
        insight_first_round = reasoner.initial_sampling()
        initial_candidates = normalize_candidates(
            candidates=reasoner.optimization_first_round(insight_first_round),
            feature_df=feature_df,
            parameter_specs=parameter_specs,
        )
        if not initial_candidates:
            raise RuntimeError(
                "initial_sampling did not yield any valid candidates in the tabular design space."
            )
        if use_llm_initial_count:
            initial_candidates = initial_candidates[:total_budget]
            if not initial_candidates:
                raise RuntimeError(
                    "No valid initial candidates remained after applying the total budget cap."
                )
        else:
            resolved_init_size = min(init_size, total_budget)
            initial_candidates = _fill_initial_candidates(
                initial_candidates=initial_candidates,
                feature_df=feature_df,
                parameter_specs=parameter_specs,
                target_count=resolved_init_size,
                seed=seed,
            )
        trial = reasoner.run_bo_experiment(experiment, initial_candidates)
        reasoner._save_experiment_data(experiment, trial)
        record_batch_observations(
            trial=trial,
            candidates=initial_candidates,
            feature_df=feature_df,
            feature_columns=feature_columns,
            selected_indices=selected_indices,
            observed_values=observed_values,
            phases=phases,
            phase="init",
        )
        current_best = float(np.max(observed_values))
        progress.update(len(initial_candidates))
        progress.set_postfix_str(f"best={current_best:.4f}")
        log(
            f"[ReasoningBO][{dataset_name.upper()}] Trial {seed + 1}: "
            f"warm start completed with {len(initial_candidates)} initial evaluations, "
            f"current_best={current_best:.4f}"
        )

        round_idx = 0
        while len(observed_values) < total_budget:
            round_idx += 1
            remaining_budget = total_budget - len(observed_values)
            remaining_candidates = len(df) - len(observed_values)
            requested_batch_size = (
                reasoning_batch_size
                if allow_batch_overshoot
                else min(reasoning_batch_size, remaining_budget)
            )
            batch_size = min(requested_batch_size, remaining_candidates)
            if batch_size <= 0:
                break
            log(
                f"[ReasoningBO][{dataset_name.upper()}] Trial {seed + 1}: "
                f"reasoning round {round_idx}, batch_size={batch_size}, "
                f"evaluated={len(observed_values)}, current_best={current_best:.4f}"
            )
            candidate_batch = reasoner.optimization_loop(
                experiment=experiment,
                trial=trial,
                bo_model=bo_model,
                n=batch_size,
            )
            candidate_batch = fill_candidates(
                candidates=candidate_batch if isinstance(candidate_batch, list) else [],
                feature_df=feature_df,
                parameter_specs=parameter_specs,
                bo_model=bo_model,
                experiment=experiment,
                n_needed=batch_size,
            )
            log(
                f"[ReasoningBO][{dataset_name.upper()}] Trial {seed + 1}: "
                f"evaluating {len(candidate_batch)} points this round"
            )
            trial = reasoner.run_bo_experiment(experiment, candidate_batch)
            record_batch_observations(
                trial=trial,
                candidates=candidate_batch,
                feature_df=feature_df,
                feature_columns=feature_columns,
                selected_indices=selected_indices,
                observed_values=observed_values,
                phases=phases,
                phase="bo",
            )
            current_best = float(np.max(observed_values))
            progress.update(len(candidate_batch))
            progress.set_postfix_str(f"best={current_best:.4f}")

    reasoner._save_experiment_data(experiment, trial)
    reasoner.generate_experiment_analysis()
    return build_trial_trace(
        selected_indices=selected_indices,
        observed_values=observed_values,
        phases=phases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reasoning-BO with Kimi on a tabular chemistry dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial-start-index", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--init-size", type=int, default=3)
    parser.add_argument("--reasoning-batch-size", type=int, default=3)
    parser.add_argument(
        "--use-llm-initial-count",
        action="store_true",
        help="Use all valid initial candidates proposed by the LLM, capped only by --total-budget.",
    )
    parser.add_argument(
        "--allow-batch-overshoot",
        action="store_true",
        help="Always evaluate a full reasoning batch even if that slightly exceeds --total-budget.",
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--exclude-columns", default=None)
    parser.add_argument("--text-columns", default=None)
    parser.add_argument("--append-to-existing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cap_cpu_threads()
    if not (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("REASONINGBO_API_KEY")
        or os.getenv("MOONSHOT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    ):
        raise RuntimeError(
            "Missing Kimi-compatible API key. Set DASHSCOPE_API_KEY "
            "(preferred), REASONINGBO_API_KEY, MOONSHOT_API_KEY, or OPENAI_API_KEY."
        )

    spec, df = load_benchmark_spec(
        ROOT,
        args.dataset,
        data_path=args.data_path,
        target_column=args.target_column,
        feature_columns=args.feature_columns,
        exclude_columns=args.exclude_columns,
        text_columns=args.text_columns,
    )
    if args.total_budget <= 0:
        raise ValueError("--total-budget must be positive.")
    if args.init_size <= 0:
        raise ValueError("--init-size must be positive.")
    if args.reasoning_batch_size <= 0:
        raise ValueError("--reasoning-batch-size must be positive.")
    if args.trial_start_index <= 0:
        raise ValueError("--trial-start-index must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative.")
    if not args.use_llm_initial_count and args.init_size > args.total_budget:
        raise ValueError("--init-size must be smaller than or equal to --total-budget.")
    if args.total_budget > len(df):
        raise ValueError(
            f"Requested total budget {args.total_budget} exceeds dataset size {len(df)}."
        )
    if args.use_llm_initial_count:
        log(
            f"[ReasoningBO][{args.dataset.upper()}] Using the full valid LLM-proposed initialization set "
            f"(capped at {args.total_budget} evaluations) instead of --init-size={args.init_size}."
        )

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "reasoning_bo" / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.dataset}_reasoning_bo_results.npz"
    requested_trial_numbers = np.arange(
        args.trial_start_index,
        args.trial_start_index + args.trials,
        dtype=int,
    )
    existing_payload = _load_existing_results(results_path) if args.append_to_existing else None
    if existing_payload is None:
        existing_results = None
        existing_trace_lengths = None
        existing_trial_numbers = np.empty(0, dtype=int)
    else:
        existing_results, existing_trace_lengths, existing_trial_numbers = existing_payload
        overlap = np.intersect1d(existing_trial_numbers, requested_trial_numbers)
        if overlap.size:
            overlap_text = ", ".join(f"trial_{trial_number:02d}" for trial_number in overlap.tolist())
            raise ValueError(f"Refusing to overwrite existing reasoning-BO trial(s): {overlap_text}")
    log(
        f"[ReasoningBO][{args.dataset.upper()}] Loaded dataset with {len(df)} rows, "
        f"total_budget={args.total_budget}, init_size={args.init_size}, "
        f"reasoning_batch_size={args.reasoning_batch_size}, "
        f"trials={args.trials}, trial_start_index={args.trial_start_index}, seed_start={args.seed_start}"
    )

    traces: list[TrialTrace] = []
    for offset, trial_number in enumerate(requested_trial_numbers.tolist()):
        seed = args.seed_start + offset
        trial_dir = output_dir / f"trial_{trial_number:02d}"
        if trial_dir.exists() and any(trial_dir.iterdir()):
            raise ValueError(f"Refusing to overwrite non-empty trial directory: {trial_dir}")
        trial_dir.mkdir(parents=True, exist_ok=True)
        traces.append(
            run_trial(
                dataset_name=args.dataset,
                df=df.copy(),
                data_path=spec.data_path,
                target_column=spec.target_column,
                feature_columns=spec.feature_columns,
                total_budget=args.total_budget,
                init_size=args.init_size,
                reasoning_batch_size=args.reasoning_batch_size,
                use_llm_initial_count=args.use_llm_initial_count,
                allow_batch_overshoot=args.allow_batch_overshoot,
                seed=seed,
                output_dir=trial_dir,
            )
        )

    new_results, new_trace_lengths = build_best_results_matrix(traces)
    if existing_results is not None and existing_trace_lengths is not None:
        combined_width = max(existing_results.shape[1], new_results.shape[1])
        results = np.concatenate(
            [
                _pad_results_matrix(existing_results, combined_width),
                _pad_results_matrix(new_results, combined_width),
            ],
            axis=0,
        )
        trace_lengths = np.concatenate([existing_trace_lengths, new_trace_lengths])
        trial_numbers = np.concatenate([existing_trial_numbers, requested_trial_numbers])
    else:
        results = new_results
        trace_lengths = new_trace_lengths
        trial_numbers = requested_trial_numbers
    np.savez(
        results_path,
        results=results,
        trace_lengths=trace_lengths,
        trial_numbers=trial_numbers,
    )
    export_trajectory_artifacts(
        output_dir=output_dir,
        stem=f"{args.dataset}_reasoning_bo",
        traces=traces,
        trial_numbers=requested_trial_numbers,
    )
    initial_values, final_values = summarize_best_results(results, trace_lengths)
    summary = {
        "dataset": args.dataset,
        "trials": int(len(trial_numbers)),
        "trial_numbers": trial_numbers.tolist(),
        "target_budget": args.total_budget,
        "actual_evaluations_per_trial": trace_lengths.tolist(),
        "init_size": args.init_size,
        "reasoning_batch_size": args.reasoning_batch_size,
        "use_llm_initial_count": args.use_llm_initial_count,
        "allow_batch_overshoot": args.allow_batch_overshoot,
        "data_path": str(spec.data_path),
        "target_column": spec.target_column,
        "feature_columns": spec.feature_columns,
        "initial_mean": float(initial_values.mean()),
        "final_mean": float(final_values.mean()),
        "final_std": float(final_values.std()),
        "llm_model": os.getenv("REASONINGBO_LLM_MODEL", os.getenv("KIMI_MODEL", "kimi-k2.5-thinking")),
    }
    (output_dir / f"{args.dataset}_reasoning_bo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(
        f"[ReasoningBO][{args.dataset.upper()}] Completed all trials. "
        f"initial_mean={summary['initial_mean']:.4f}, final_mean={summary['final_mean']:.4f}, final_std={summary['final_std']:.4f}"
    )


if __name__ == "__main__":
    main()
