from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gpytorch.priors import GammaPrior
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "baseline"))

from common.tabular_benchmarks import dataframe_to_one_hot, load_benchmark_spec
from kimi_client import call_kimi_chat, parse_jsonish_response
from model import GP_Model, compute_probability, train_preference
from acquisition import acquire, optim


DEFAULT_INIT_SIZE = 1
DEFAULT_MAX_QUESTIONS = {"dar": 2000, "ocm": 20000}
SURVEY_PROGRESS_EVERY = 100
SURVEY_MAX_ATTEMPTS = 3
DIAGNOSTIC_LOG_PATH: Path | None = None
SIGNAL_LOG_LOCK = threading.Lock()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_diagnostic_log(message: str) -> None:
    global DIAGNOSTIC_LOG_PATH
    if DIAGNOSTIC_LOG_PATH is None:
        return
    DIAGNOSTIC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SIGNAL_LOG_LOCK:
        with DIAGNOSTIC_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


def safe_process_snapshot() -> str:
    try:
        proc = subprocess.run(
            ["ps", "-o", "pid,ppid,pgid,sid,stat,etime,cmd", "-p", str(os.getpid())],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"<failed to collect process snapshot: {type(exc).__name__}: {exc}>"


def install_signal_diagnostics(output_dir: Path) -> None:
    global DIAGNOSTIC_LOG_PATH
    DIAGNOSTIC_LOG_PATH = output_dir / "prefbo_signal_diagnostics.log"
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    def _log_signal(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        stack_text = "".join(traceback.format_stack())
        payload = (
            f"[PrefBO][signal] pid={os.getpid()} received {signame} ({signum})\n"
            f"[PrefBO][signal] process snapshot:\n{safe_process_snapshot()}\n"
            f"[PrefBO][signal] python stack:\n{stack_text}"
        )
        log(payload)
        write_diagnostic_log(payload)

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]:
        previous_handler = signal.getsignal(sig)

        def _handler(signum: int, frame: Any, *, _prev=previous_handler) -> None:
            _log_signal(signum, frame)
            if callable(_prev):
                _prev(signum, frame)
                return
            if _prev == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
                return
            if _prev == signal.SIG_IGN:
                return

        signal.signal(sig, _handler)

    def _on_exit() -> None:
        write_diagnostic_log(f"[PrefBO][exit] pid={os.getpid()} exiting normally")

    atexit.register(_on_exit)


def cap_cpu_threads(max_threads: int = 100) -> int:
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


def load_existing_results(results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not results_path.exists():
        return None
    with np.load(results_path, allow_pickle=True) as payload:
        results = np.asarray(payload["results"], dtype=float)
        preference_scores = np.asarray(payload["preference_scores"], dtype=float)
        comparisons = np.asarray(payload["comparisons"])
        if "trial_numbers" in payload.files:
            trial_numbers = np.asarray(payload["trial_numbers"], dtype=int)
        else:
            trial_numbers = np.arange(1, results.shape[0] + 1, dtype=int)
    return results, preference_scores, comparisons, trial_numbers


def validate_comparisons(comparisons: np.ndarray, n_rows: int, source: Path) -> np.ndarray:
    comparisons = np.asarray(comparisons)
    if comparisons.ndim != 2 or comparisons.shape[1] != 2:
        raise ValueError(
            f"Comparison data from {source} must have shape (n, 2); got {comparisons.shape}."
        )
    comparisons = comparisons.astype(int, copy=False)
    if comparisons.size == 0:
        return comparisons.reshape(0, 2)
    if comparisons.min() < 0 or comparisons.max() >= n_rows:
        raise ValueError(
            f"Comparison indices from {source} must fall in [0, {n_rows - 1}]."
        )
    return comparisons


def validate_preference_scores(
    preference_scores: np.ndarray,
    n_rows: int,
    source: Path,
) -> np.ndarray:
    preference_scores = np.asarray(preference_scores, dtype=float).reshape(-1)
    if preference_scores.shape[0] != n_rows:
        raise ValueError(
            f"Preference scores from {source} must have length {n_rows}; "
            f"got {preference_scores.shape[0]}."
        )
    if not np.isfinite(preference_scores).all():
        raise ValueError(f"Preference scores from {source} contain non-finite values.")
    return preference_scores


def load_external_preferences(
    preferences_path: Path,
    n_rows: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    source = preferences_path.resolve()
    suffix = source.suffix.lower()

    if suffix == ".npy":
        payload = np.load(source, allow_pickle=True)
        if getattr(payload, "shape", ()) == ():
            payload = payload.item()
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Unsupported scalar payload type in {source}: {type(payload).__name__}."
                )
            raw_scores = payload.get("preference_scores")
            raw_comparisons = payload.get("comparisons")
        else:
            raw_scores = payload if getattr(payload, "ndim", 0) == 1 else None
            raw_comparisons = payload if getattr(payload, "ndim", 0) == 2 else None
    elif suffix == ".csv":
        df = pd.read_csv(source)
        columns = {str(col).strip() for col in df.columns}
        if {"winner_idx", "loser_idx"}.issubset(columns):
            raw_comparisons = df.loc[:, ["winner_idx", "loser_idx"]].to_numpy(dtype=int)
            raw_scores = None
        elif {"idx", "preference_score"}.issubset(columns):
            ordered = df.sort_values("idx")
            raw_scores = ordered["preference_score"].to_numpy(dtype=float)
            raw_comparisons = None
        elif df.shape[1] == 1:
            raw_scores = df.iloc[:, 0].to_numpy(dtype=float)
            raw_comparisons = None
        elif df.shape[1] == 2:
            raw_comparisons = df.iloc[:, :2].to_numpy(dtype=int)
            raw_scores = None
        else:
            raise ValueError(
                f"Unsupported CSV format in {source}. Expected winner/loser columns "
                "or per-row preference scores."
            )
    else:
        raise ValueError(
            f"Unsupported preference file format: {source}. Expected .npy or .csv."
        )

    preference_scores = (
        validate_preference_scores(raw_scores, n_rows=n_rows, source=source)
        if raw_scores is not None
        else None
    )
    comparisons = (
        validate_comparisons(raw_comparisons, n_rows=n_rows, source=source)
        if raw_comparisons is not None
        else None
    )
    if preference_scores is None and comparisons is None:
        raise ValueError(f"No usable preference data found in {source}.")
    return preference_scores, comparisons, str(source)


def compare_optional_arrays(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    float_array: bool,
) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if left.shape != right.shape:
        return False
    if float_array:
        return np.allclose(left, right)
    return np.array_equal(left, right)


def log(message: str) -> None:
    print(message, flush=True)


def estimate_prompt_tokens(text: str) -> int:
    # Rough heuristic for monitoring only; avoids adding tokenizer dependencies.
    return max(1, int(round(len(text) / 3.6)))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_survey_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def load_existing_survey(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "question" in df.columns:
        df["question"] = df["question"].astype(int)
    return df


def comparison_from_row(row: pd.Series | dict[str, Any]) -> list[int]:
    predicted_setup = str(row["pred_setup"]).strip().upper()
    idx_a = int(row["idx_a"])
    idx_b = int(row["idx_b"])
    return [idx_a, idx_b] if predicted_setup == "A" else [idx_b, idx_a]


def build_prompt(dataset_name: str, row_a: pd.Series, row_b: pd.Series, context_df: pd.DataFrame | None) -> str:
    if dataset_name == "dar":
        context_block = ""
        if context_df is not None and len(context_df):
            lines = [
                f"Base={row['base_SMILES']}; Ligand={row['ligand_SMILES']}; Solvent={row['solvent_SMILES']}; "
                f"Concentration={row['concentration']}; Temperature={row['temperature']}; Yield={row['yield']}"
                for _, row in context_df.iterrows()
            ]
            context_block = "Known examples:\n" + "\n".join(lines[:5]) + "\n\n"
        return (
            "You are an expert chemist predicting direct arylation yield.\n"
            f"{context_block}"
            "Choose which setup should give the higher yield. Return valid JSON with keys Setup and reasoning.\n\n"
            "Setup A:\n"
            f"Base: {row_a['base_SMILES']}\n"
            f"Ligand: {row_a['ligand_SMILES']}\n"
            f"Solvent: {row_a['solvent_SMILES']}\n"
            f"Concentration: {row_a['concentration']}\n"
            f"Temperature: {row_a['temperature']}\n\n"
            "Setup B:\n"
            f"Base: {row_b['base_SMILES']}\n"
            f"Ligand: {row_b['ligand_SMILES']}\n"
            f"Solvent: {row_b['solvent_SMILES']}\n"
            f"Concentration: {row_b['concentration']}\n"
            f"Temperature: {row_b['temperature']}\n"
        )
    if dataset_name == "ocm":
        context_block = ""
        if context_df is not None and len(context_df):
            lines = [
                f"Catalyst={row['Name']}; Support={row['Support']}; Temp={row['Temp']}; "
                f"Ar={row['Ar_flow']}; CH4={row['CH4_flow']}; O2={row['O2_flow']}; CT={row['CT']}; "
                f"Performance={row['Performance']}"
                for _, row in context_df.iterrows()
            ]
            context_block = "Known examples:\n" + "\n".join(lines[:5]) + "\n\n"
        return (
            "You are an expert catalysis scientist predicting oxidative coupling of methane performance.\n"
            f"{context_block}"
            "Choose which setup should give the higher performance. Return valid JSON with keys Setup and reasoning.\n\n"
            "Setup A:\n"
            f"Catalyst: {row_a['Name']} on {row_a['Support']}\n"
            f"M1/M2/M3: {row_a['M1']}/{row_a['M2']}/{row_a['M3']}\n"
            f"M1_mol/M2_mol/M3_mol: {row_a['M1_mol']}/{row_a['M2_mol']}/{row_a['M3_mol']}\n"
            f"Temp: {row_a['Temp']}\n"
            f"Ar_flow: {row_a['Ar_flow']}\n"
            f"CH4_flow: {row_a['CH4_flow']}\n"
            f"O2_flow: {row_a['O2_flow']}\n"
            f"CT: {row_a['CT']}\n\n"
            "Setup B:\n"
            f"Catalyst: {row_b['Name']} on {row_b['Support']}\n"
            f"M1/M2/M3: {row_b['M1']}/{row_b['M2']}/{row_b['M3']}\n"
            f"M1_mol/M2_mol/M3_mol: {row_b['M1_mol']}/{row_b['M2_mol']}/{row_b['M3_mol']}\n"
            f"Temp: {row_b['Temp']}\n"
            f"Ar_flow: {row_b['Ar_flow']}\n"
            f"CH4_flow: {row_b['CH4_flow']}\n"
            f"O2_flow: {row_b['O2_flow']}\n"
            f"CT: {row_b['CT']}\n"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def generate_question_pairs(n_rows: int, n_questions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_questions):
        idx_a, idx_b = rng.choice(n_rows, size=2, replace=False)
        pairs.append([int(idx_a), int(idx_b)])
    return np.array(pairs, dtype=int)


def get_default_questions_file(dataset_name: str) -> Path:
    return ROOT / "baseline" / "Pref-BO" / "questions" / f"{dataset_name.lower()}_questions_seed0.npy"


def load_or_create_questions(
    dataset_name: str,
    n_rows: int,
    questions_file: Path | None,
    seed: int,
) -> tuple[np.ndarray, Path]:
    dataset_key = dataset_name.lower()
    default_count = 5000 if dataset_key == "dar" else 20000
    resolved_path = Path(questions_file) if questions_file is not None else get_default_questions_file(dataset_key)
    if resolved_path.exists():
        questions = np.load(resolved_path, allow_pickle=False)
        return np.asarray(questions, dtype=int), resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    questions = generate_question_pairs(n_rows=n_rows, n_questions=default_count, seed=seed)
    np.save(resolved_path, questions)
    return questions, resolved_path


def run_llm_survey(dataset_name: str, df: pd.DataFrame, questions: np.ndarray, output_dir: Path) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_df = df.head(5).copy()
    survey_csv = output_dir / "llm_survey.csv"
    raw_jsonl = output_dir / "llm_survey_raw.jsonl"
    failure_jsonl = output_dir / "llm_survey_failures.jsonl"
    existing_df = load_existing_survey(survey_csv)
    answered_by_question: dict[int, dict[str, Any]] = {}
    for _, row in existing_df.iterrows():
        answered_by_question[int(row["question"])] = row.to_dict()
    comparisons = [comparison_from_row(row) for _, row in existing_df.iterrows()]
    cumulative_input_tokens = int(existing_df.get("prompt_tokens_est", pd.Series(dtype=float)).fillna(0).sum()) if not existing_df.empty else 0
    cumulative_output_tokens = int(existing_df.get("response_tokens_est", pd.Series(dtype=float)).fillna(0).sum()) if not existing_df.empty else 0
    api_calls = len(existing_df)
    log(
        f"[PrefBO][{dataset_name.upper()}] Starting LLM survey with {len(questions)} pairwise questions. "
        f"Intermediate CSV will be written to {survey_csv}"
    )
    if answered_by_question:
        log(
            f"[PrefBO][{dataset_name.upper()}] Resuming survey with "
            f"{len(answered_by_question)} existing answers already recorded."
        )
    for q_idx, (idx_a, idx_b) in enumerate(questions):
        if q_idx in answered_by_question:
            if (q_idx + 1) % SURVEY_PROGRESS_EVERY == 0 or q_idx == len(questions) - 1:
                log(
                    f"[PrefBO][{dataset_name.upper()}] Survey progress: "
                    f"{q_idx + 1}/{len(questions)} questions, api_calls={api_calls}, "
                    f"est_input_tokens={cumulative_input_tokens}, "
                    f"est_output_tokens={cumulative_output_tokens}"
                )
            continue
        log(
            f"[PrefBO][{dataset_name.upper()}] Survey question {q_idx + 1}/{len(questions)}: "
            f"compare idx_a={int(idx_a)} vs idx_b={int(idx_b)}"
        )
        prompt = build_prompt(dataset_name, df.iloc[idx_a], df.iloc[idx_b], context_df)
        prompt_tokens = estimate_prompt_tokens(prompt)
        success = False
        for attempt in range(1, SURVEY_MAX_ATTEMPTS + 1):
            raw = call_kimi_chat(prompt)
            response_tokens = estimate_prompt_tokens(raw)
            api_calls += 1
            cumulative_input_tokens += prompt_tokens
            cumulative_output_tokens += response_tokens
            append_jsonl(
                raw_jsonl,
                {
                    "question": int(q_idx),
                    "attempt": attempt,
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "prompt": prompt,
                    "raw_response": raw,
                    "prompt_tokens_est": prompt_tokens,
                    "response_tokens_est": response_tokens,
                },
            )
            try:
                parsed = parse_jsonish_response(raw)
                predicted_setup = str(parsed.get("Setup", "A")).strip().upper()
                if predicted_setup not in {"A", "B"}:
                    predicted_setup = "A"
                row = {
                    "question": q_idx,
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "pred_setup": predicted_setup,
                    "raw_response": raw,
                    "prompt_tokens_est": prompt_tokens,
                    "response_tokens_est": response_tokens,
                    "reasoning": parsed.get("reasoning", ""),
                    "status": "success",
                    "attempts_used": attempt,
                }
                append_survey_row(survey_csv, row)
                answered_by_question[q_idx] = row
                comparisons.append(comparison_from_row(row))
                log(
                    f"[PrefBO][{dataset_name.upper()}] Survey question {q_idx + 1}/{len(questions)} answered: "
                    f"pred_setup={predicted_setup}"
                )
                success = True
                break
            except Exception as exc:
                append_jsonl(
                    failure_jsonl,
                    {
                        "question": int(q_idx),
                        "attempt": attempt,
                        "idx_a": int(idx_a),
                        "idx_b": int(idx_b),
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": raw,
                    },
                )
                if attempt < SURVEY_MAX_ATTEMPTS:
                    log(
                        f"[PrefBO][{dataset_name.upper()}] Survey question {q_idx + 1}/{len(questions)} "
                        f"parse failed on attempt {attempt}/{SURVEY_MAX_ATTEMPTS}; retrying"
                    )
        if not success:
            log(
                f"[PrefBO][{dataset_name.upper()}] Survey question {q_idx + 1}/{len(questions)} "
                "failed after all retries; leaving it unanswered for a later resume"
            )
        completed = q_idx + 1
        if completed % SURVEY_PROGRESS_EVERY == 0 or completed == len(questions):
            log(
                f"[PrefBO][{dataset_name.upper()}] Survey progress: "
                f"{completed}/{len(questions)} questions, api_calls={api_calls}, "
                f"est_input_tokens={cumulative_input_tokens}, "
                f"est_output_tokens={cumulative_output_tokens}"
            )
    if survey_csv.exists():
        deduped = load_existing_survey(survey_csv).sort_values("question").drop_duplicates(
            subset=["question"], keep="last"
        )
        deduped.to_csv(survey_csv, index=False)
    log(f"[PrefBO][{dataset_name.upper()}] LLM survey finished.")
    return np.array(comparisons, dtype=int)


def run_trial(
    X: np.ndarray,
    y: np.ndarray,
    preference_scores: np.ndarray,
    init_size: int,
    total_budget: int,
    seed: int,
):
    set_global_seed(seed)
    log(
        f"[PrefBO] Trial seed={seed} starting with init_size={init_size}, total_budget={total_budget}"
    )
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(X))
    top_pref_idx = np.argsort(preference_scores)[-max(init_size * 3, init_size):]
    done_idx = rng.choice(top_pref_idx, size=init_size, replace=False)
    remaining_idx = np.array([idx for idx in all_idx if idx not in set(done_idx.tolist())])

    best_trace = [float(np.max(y[done_idx]))]
    log(
        f"[PrefBO] Trial seed={seed} initialized with indices={done_idx.tolist()} "
        f"best={best_trace[-1]:.4f}"
    )
    bo_steps = total_budget - init_size

    for step in range(bo_steps):
        log(
            f"[PrefBO] Trial seed={seed} iteration {step + 1}/{bo_steps}: "
            f"evaluated={len(done_idx)}, remaining={len(remaining_idx)}, current_best={best_trace[-1]:.4f}"
        )
        scaler = StandardScaler()
        y_train = y[done_idx].reshape(-1, 1)
        if step > 0:
            y_train = scaler.fit_transform(y_train).flatten()
        else:
            y_train = y_train.flatten()

        x_train = torch.tensor(X[done_idx], dtype=torch.float64)
        y_train_t = torch.tensor(y_train, dtype=torch.float64)
        surrogate = GP_Model(
            x_train,
            y_train_t,
            gpu=False,
            nu=2.5,
            noise_constraint=1e-5,
            lengthscale_prior=[GammaPrior(2.0, 0.2), 5.0],
            outputscale_prior=[GammaPrior(5.0, 0.5), 8.0],
            noise_prior=[GammaPrior(1.5, 0.5), 1.0],
            n_restarts=0,
            learning_rate=0.1,
            training_iters=100,
        )
        surrogate.fit()

        opt = optim(
            surrogate=surrogate,
            input_space=torch.tensor(X[remaining_idx], dtype=torch.float64),
            method="pibo",
            preference=preference_scores[remaining_idx],
        )

        class Args:
            def __init__(self, fmax: float, iteration: int):
                self.fmax = fmax
                self.beta = 10.0
                self.iter = iteration

        args = Args(float(np.max(y_train if step > 0 else y[done_idx])), step + 1)
        chosen_local_idx = acquire(opt, args)
        chosen_idx = int(remaining_idx[chosen_local_idx])
        remaining_idx = np.delete(remaining_idx, chosen_local_idx)
        done_idx = np.append(done_idx, chosen_idx)
        best_trace.append(float(np.max(y[done_idx])))
        log(
            f"[PrefBO] Trial seed={seed} selected idx={chosen_idx} "
            f"observed={float(y[chosen_idx]):.4f} new_best={best_trace[-1]:.4f}"
        )

    log(f"[PrefBO] Trial seed={seed} finished with final_best={best_trace[-1]:.4f}")
    return np.array(best_trace, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Run preference BO on DAR or OCM with Kimi.")
    parser.add_argument("--dataset", choices=["dar", "ocm"], required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial-start-index", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--init-size", type=int, default=None)
    parser.add_argument("--preferences-file", default=None)
    parser.add_argument("--questions-file", default=None)
    parser.add_argument("--question-seed", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-cpu-threads", type=int, default=50)
    parser.add_argument("--append-to-existing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.max_cpu_threads <= 0:
        raise ValueError("--max-cpu-threads must be positive.")
    thread_cap = cap_cpu_threads(args.max_cpu_threads)
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
    requires_llm = args.preferences_file is None
    if requires_llm and not (
        os.getenv("PREFBO_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("MOONSHOT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    ):
        raise RuntimeError("Missing Kimi-compatible API key. Set DASHSCOPE_API_KEY, MOONSHOT_API_KEY, PREFBO_API_KEY, or OPENAI_API_KEY.")

    spec, df = load_benchmark_spec(ROOT, args.dataset)
    if args.total_budget > len(df):
        raise ValueError(f"--total-budget={args.total_budget} exceeds dataset size {len(df)}.")
    max_questions = args.max_questions or DEFAULT_MAX_QUESTIONS[args.dataset]
    if max_questions <= 0:
        raise ValueError("--max-questions must be positive.")
    log(
        f"[PrefBO][{args.dataset.upper()}] Loaded dataset with {len(df)} rows, "
        f"budget={args.total_budget}, init_size={resolved_init_size}, trials={args.trials}, "
        f"max_questions={max_questions}, cpu_threads={thread_cap}"
    )
    X_df = dataframe_to_one_hot(df, spec.feature_columns)
    X = X_df.to_numpy(dtype=float)
    y = df[spec.target_column].to_numpy(dtype=float)

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "prefbo" / args.dataset)
    install_signal_diagnostics(output_dir)
    results_path = output_dir / f"{args.dataset}_prefbo_results.npz"
    requested_trial_numbers = np.arange(
        args.trial_start_index,
        args.trial_start_index + args.trials,
        dtype=int,
    )
    existing_payload = load_existing_results(results_path) if args.append_to_existing else None
    if existing_payload is None:
        existing_results = None
        existing_preference_scores = None
        existing_comparisons = None
        existing_trial_numbers = np.empty(0, dtype=int)
    else:
        (
            existing_results,
            existing_preference_scores,
            existing_comparisons,
            existing_trial_numbers,
        ) = existing_payload
        overlap = np.intersect1d(existing_trial_numbers, requested_trial_numbers)
        if overlap.size:
            overlap_text = ", ".join(f"trial_{trial_number:02d}" for trial_number in overlap.tolist())
            raise ValueError(f"Refusing to overwrite existing PrefBO trial(s): {overlap_text}")
    survey_dir = output_dir / "survey"
    log(f"[PrefBO][{args.dataset.upper()}] Writing outputs to {output_dir}")
    preference_source = "existing results cache"
    if args.preferences_file:
        preference_scores, comparisons, preference_source = load_external_preferences(
            Path(args.preferences_file),
            n_rows=len(df),
        )
        log(
            f"[PrefBO][{args.dataset.upper()}] Loaded external preference data from {preference_source}"
        )
        if existing_payload is not None:
            cached_scores = existing_preference_scores
            cached_comparisons = existing_comparisons
            if comparisons is not None and preference_scores is None and compare_optional_arrays(
                cached_comparisons,
                comparisons,
                float_array=False,
            ):
                preference_scores = cached_scores
            if preference_scores is not None and comparisons is None and compare_optional_arrays(
                cached_scores,
                preference_scores,
                float_array=True,
            ):
                comparisons = cached_comparisons

            scores_match = (
                True
                if preference_scores is None
                else compare_optional_arrays(cached_scores, preference_scores, float_array=True)
            )
            comparisons_match = (
                True
                if comparisons is None
                else compare_optional_arrays(cached_comparisons, comparisons, float_array=False)
            )
            if not scores_match or not comparisons_match:
                raise ValueError(
                    "Existing results file contains different preference data than "
                    f"--preferences-file={args.preferences_file}. Use a fresh output directory "
                    "or remove the incompatible cache first."
                )
    elif existing_preference_scores is not None and existing_comparisons is not None:
        preference_scores = existing_preference_scores
        comparisons = existing_comparisons
        log(
            f"[PrefBO][{args.dataset.upper()}] Reusing existing comparisons "
            f"({len(comparisons)}) and preference scores from {results_path}"
        )
    else:
        questions, questions_path = load_or_create_questions(
            dataset_name=args.dataset,
            n_rows=len(df),
            questions_file=Path(args.questions_file) if args.questions_file else None,
            seed=args.question_seed,
        )
        questions = questions[:max_questions]
        if len(questions) == 0:
            raise RuntimeError("No survey questions available after applying max_questions.")
        log(
            f"[PrefBO][{args.dataset.upper()}] Using question file {questions_path} "
            f"with {len(questions)} questions after truncation"
        )
        comparisons = run_llm_survey(args.dataset, df, questions, survey_dir)

        X_pref = torch.tensor(X, dtype=torch.float64)
        log(f"[PrefBO][{args.dataset.upper()}] Training preference model on {len(comparisons)} comparisons")
        pref_model = train_preference(x_train=X_pref, train_comp=comparisons)
        preference_scores = compute_probability(model=pref_model, all_x=X)
        log(f"[PrefBO][{args.dataset.upper()}] Preference scores computed for {len(preference_scores)} candidates")

    if preference_scores is None:
        if comparisons is None:
            raise RuntimeError("Preference BO requires either preference scores or pairwise comparisons.")
        X_pref = torch.tensor(X, dtype=torch.float64)
        log(
            f"[PrefBO][{args.dataset.upper()}] Training preference model on "
            f"{len(comparisons)} externally supplied comparisons"
        )
        pref_model = train_preference(x_train=X_pref, train_comp=comparisons)
        preference_scores = compute_probability(model=pref_model, all_x=X)
        log(f"[PrefBO][{args.dataset.upper()}] Preference scores computed for {len(preference_scores)} candidates")

    if comparisons is None:
        comparisons = np.empty((0, 2), dtype=int)

    trial_seeds = []
    traces = []
    for offset in range(args.trials):
        seed = args.seed_start + offset
        trial_seeds.append(seed)
        traces.append(
            run_trial(
                X=X,
                y=y,
                preference_scores=preference_scores,
                init_size=resolved_init_size,
                total_budget=args.total_budget,
                seed=seed,
            )
        )

    new_results = np.vstack(traces)
    if existing_results is not None:
        results = np.vstack([existing_results, new_results])
        trial_numbers = np.concatenate([existing_trial_numbers, requested_trial_numbers])
    else:
        results = new_results
        trial_numbers = requested_trial_numbers
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_path,
        results=results,
        preference_scores=preference_scores,
        comparisons=comparisons,
        trial_numbers=trial_numbers,
    )
    summary = {
        "dataset": args.dataset,
        "trials": int(len(trial_numbers)),
        "trial_numbers": trial_numbers.tolist(),
        "requested_trial_seeds": trial_seeds,
        "seed_start": args.seed_start,
        "total_budget": args.total_budget,
        "init_size": resolved_init_size,
        "preference_source": preference_source,
        "cpu_thread_cap": thread_cap,
        "initial_mean": float(results[:, 0].mean()),
        "final_mean": float(results[:, -1].mean()),
        "final_std": float(results[:, -1].std()),
    }
    (output_dir / f"{args.dataset}_prefbo_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    log(
        f"[PrefBO][{args.dataset.upper()}] Completed all trials. "
        f"initial_mean={summary['initial_mean']:.4f}, final_mean={summary['final_mean']:.4f}, final_std={summary['final_std']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        payload = (
            f"[PrefBO][KeyboardInterrupt] pid={os.getpid()} parent={os.getppid()}\n"
            f"[PrefBO][KeyboardInterrupt] process snapshot:\n{safe_process_snapshot()}\n"
            f"[PrefBO][KeyboardInterrupt] traceback:\n{traceback.format_exc()}"
        )
        log(payload)
        write_diagnostic_log(payload)
        raise
