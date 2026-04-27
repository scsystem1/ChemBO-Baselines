from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "baseline"))

from common.tabular_benchmarks import dataframe_to_texts, load_benchmark_spec
from boicl.asktell import AskTellFewShotTopk
from boicl.pool import Pool


DEFAULT_INIT_SIZE = 2


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
        os.environ[env_name] = str(thread_cap)
    try:
        torch.set_num_threads(thread_cap)
    except Exception:
        pass
    return thread_cap


def log(message: str) -> None:
    print(message, flush=True)


SYSTEM_PROMPTS = {
    "dar": (
        "You are an expert chemist modeling direct arylation yield. "
        "Given a reaction condition description, answer with only one numeric yield prediction."
    ),
    "ocm": (
        "You are an expert catalysis scientist modeling oxidative coupling of methane performance. "
        "Given a catalyst and condition description, answer with only one numeric performance prediction."
    ),
}


def run_trial(
    dataset_name: str,
    texts: list[str],
    y: np.ndarray,
    total_budget: int,
    init_size: int,
    seed: int,
    model_name: str,
):
    log(f"[BO-ICL][{dataset_name.upper()}] Trial {seed + 1} starting with init_size={init_size}, total_budget={total_budget}")
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(texts))
    init_idx = rng.choice(all_idx, size=init_size, replace=False)
    remaining = np.array([idx for idx in all_idx if idx not in set(init_idx.tolist())])

    optimizer = AskTellFewShotTopk(
        model=model_name,
        selector_k=None,
        k=5,
        temperature=1.0,
        use_logprobs=False,
        x_name="experiment description",
        y_name="target",
    )

    selected = list(init_idx.tolist())
    for idx in init_idx:
        optimizer.tell(texts[idx], float(y[idx]))

    cumulative_best = [float(np.max(y[selected]))]
    log(
        f"[BO-ICL][{dataset_name.upper()}] Trial {seed + 1} initialized with indices={init_idx.tolist()} "
        f"best={cumulative_best[-1]:.4f}"
    )
    system_message = SYSTEM_PROMPTS[dataset_name]

    for step in range(total_budget - init_size):
        pool_items = [texts[idx] for idx in remaining]
        log(
            f"[BO-ICL][{dataset_name.upper()}] Trial {seed + 1} iteration {step + 1}/{total_budget - init_size}: "
            f"evaluated={len(selected)}, remaining={len(remaining)}, current_best={cumulative_best[-1]:.4f}"
        )
        pool = Pool(pool_items)
        next_text, _, _ = optimizer.ask(
            pool,
            aq_fxn="upper_confidence_bound",
            k=1,
            inv_filter=len(pool_items),
            system_message=system_message,
        )
        chosen_text = next_text[0]
        chosen_idx = remaining[pool_items.index(chosen_text)]
        optimizer.tell(texts[chosen_idx], float(y[chosen_idx]))
        selected.append(int(chosen_idx))
        remaining = remaining[remaining != chosen_idx]
        cumulative_best.append(float(np.max(y[selected])))
        log(
            f"[BO-ICL][{dataset_name.upper()}] Trial {seed + 1} selected idx={int(chosen_idx)} "
            f"observed={float(y[chosen_idx]):.4f} new_best={cumulative_best[-1]:.4f}"
        )

    log(f"[BO-ICL][{dataset_name.upper()}] Trial {seed + 1} finished with final_best={cumulative_best[-1]:.4f}")
    return np.array(cumulative_best, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Run BO-ICL on DAR or OCM.")
    parser.add_argument("--dataset", choices=["dar", "ocm"], required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--init-size", type=int, default=None)
    parser.add_argument("--model", default="kimi-k2.5-thinking")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cap_cpu_threads()
    resolved_init_size = args.init_size or DEFAULT_INIT_SIZE
    if args.total_budget <= 0:
        raise ValueError("--total-budget must be positive.")
    if resolved_init_size <= 0:
        raise ValueError("--init-size must be positive.")
    if resolved_init_size >= args.total_budget:
        raise ValueError("--init-size must be smaller than --total-budget.")
    if not (
        os.getenv("BOICL_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("MOONSHOT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    ):
        raise RuntimeError("Missing Kimi-compatible API key. Set DASHSCOPE_API_KEY, MOONSHOT_API_KEY, BOICL_API_KEY, or OPENAI_API_KEY.")

    spec, df = load_benchmark_spec(ROOT, args.dataset)
    if args.total_budget > len(df):
        raise ValueError(f"--total-budget={args.total_budget} exceeds dataset size {len(df)}.")
    texts = dataframe_to_texts(args.dataset, df)
    y = df[spec.target_column].to_numpy(dtype=float)

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "boicl" / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"[BO-ICL][{args.dataset.upper()}] Loaded dataset with {len(df)} rows, "
        f"budget={args.total_budget}, init_size={resolved_init_size}, trials={args.trials}, model={args.model}"
    )
    log(f"[BO-ICL][{args.dataset.upper()}] Writing outputs to {output_dir}")

    all_runs = []
    for seed in range(args.trials):
        all_runs.append(
            run_trial(
                dataset_name=args.dataset,
                texts=texts,
                y=y,
                total_budget=args.total_budget,
                init_size=resolved_init_size,
                seed=seed,
                model_name=args.model,
            )
        )

    results = np.vstack(all_runs)
    np.savez(
        output_dir / f"{args.dataset}_boicl_results.npz",
        results=results,
        dataset=args.dataset,
        total_budget=args.total_budget,
        init_size=resolved_init_size,
        model=args.model,
    )

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "trials": args.trials,
        "total_budget": args.total_budget,
        "init_size": resolved_init_size,
        "initial_mean": float(results[:, 0].mean()),
        "final_mean": float(results[:, -1].mean()),
        "final_std": float(results[:, -1].std()),
    }
    (output_dir / f"{args.dataset}_boicl_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    log(
        f"[BO-ICL][{args.dataset.upper()}] Completed all trials. "
        f"initial_mean={summary['initial_mean']:.4f}, final_mean={summary['final_mean']:.4f}, final_std={summary['final_std']:.4f}"
    )


if __name__ == "__main__":
    main()
