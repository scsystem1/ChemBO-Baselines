from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import seed_everything
from huggingface_hub import snapshot_download
from botorch.exceptions import InputDataWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "baseline"))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from common.tabular_benchmarks import dataframe_to_texts, load_benchmark_spec
from gollum.bo.optimizer import BotorchOptimizer
from gollum.data.module import BaseDataModule
from gollum.surrogate_models.gp import SurrogateModel
from gollum.utils.config import instantiate_class
from botorch.acquisition import AcquisitionFunction

DEFAULT_INIT_SIZE = 10
DEFAULT_HF_HOME = Path("/data/shared/huggingface")
DEFAULT_VISIBLE_GPUS = "2"

warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings(
    "ignore",
    message="ExpectedImprovement has known numerical issues that lead to suboptimal optimization performance",
)


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


def load_existing_results(results_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not results_path.exists():
        return None
    with np.load(results_path) as payload:
        results = np.asarray(payload["results"], dtype=float)
        if "trial_numbers" in payload.files:
            trial_numbers = np.asarray(payload["trial_numbers"], dtype=int)
        else:
            trial_numbers = np.arange(1, results.shape[0] + 1, dtype=int)
    return results, trial_numbers


def log(message: str) -> None:
    print(message, flush=True)


def to_1d_int_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=int).reshape(-1)


def configure_runtime_paths() -> Path:
    hf_home = Path(os.getenv("HF_HOME", str(DEFAULT_HF_HOME)))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("GOLLUM_CUDA_VISIBLE_DEVICES", DEFAULT_VISIBLE_GPUS))
    log(
        f"[GOLLuM] Runtime paths configured: HF_HOME={hf_home}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )
    return hf_home


def ensure_t5_available(hf_home: Path) -> None:
    hub_dir = hf_home / "hub"
    reusable_candidates = [
        hub_dir / "models--t5-base",
        hub_dir / "models--google--t5-base",
        hub_dir / "models--google-t5--t5-base",
    ]
    if any(path.exists() for path in reusable_candidates):
        log(f"[GOLLuM] Reusing cached T5 model from {hub_dir}")
        return
    log(f"[GOLLuM] T5 cache not found under {hub_dir}. Downloading from Hugging Face...")
    snapshot_download(repo_id="t5-base", cache_dir=str(hf_home / "hub"))
    log("[GOLLuM] T5 download completed.")


def build_processed_dataset(dataset_name: str, output_dir: Path) -> Path:
    spec, df = load_benchmark_spec(ROOT, dataset_name)
    processed = pd.DataFrame(
        {
            "procedure": dataframe_to_texts(dataset_name, df),
            "objective": df[spec.target_column].astype(float),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_gollum_dataset.csv"
    processed.to_csv(output_path, index=False)
    log(f"[GOLLuM][{dataset_name.upper()}] Prepared dataset written to {output_path}")
    return output_path


def build_config(data_path: Path, seed: int, init_size: int, n_iters: int) -> dict:
    return {
        "data": {
            "class_path": "gollum.data.module.BaseDataModule",
            "init_args": {
                "data_path": str(data_path),
                "input_column": "procedure",
                "target_column": "objective",
                "maximize": True,
                "exclude_top": False,
                "normalize_input": "original",
                "featurizer": {
                    "class_path": "gollum.data.module.Featurizer",
                    "init_args": {
                        "representation": "get_tokens",
                        "model_name": "t5-base",
                        "pooling_method": "average",
                    },
                },
                "initializer": {
                    "class_path": "gollum.initialization.initializers.BOInitializer",
                    "init_args": {
                        "method": "true_random",
                        "n_clusters": init_size,
                        "seed": seed,
                    },
                },
            },
        },
        "bo": {
            "class_path": "gollum.bo.optimizer.BotorchOptimizer",
            "init_args": {
                "batch_strategy": "kriging",
                "batch_size": 1,
            },
        },
        "acquisition": {
            "class_path": "botorch.acquisition.analytic.ExpectedImprovement",
            "init_args": {
                "maximize": True,
            },
        },
        "surrogate_model": {
            "class_path": "gollum.surrogate_models.gp.DeepGP",
            "init_args": {
                "likelihood": {
                    "class_path": "gpytorch.likelihoods.GaussianLikelihood",
                    "init_args": {},
                },
                "covar_module": {
                    "class_path": "gpytorch.kernels.ScaleKernel",
                    "init_args": {
                        "base_kernel": {
                            "class_path": "gpytorch.kernels.MaternKernel",
                            "init_args": {
                                "nu": 2.5,
                            },
                        },
                    },
                },
                "mean_module": {
                    "class_path": "gpytorch.means.ConstantMean",
                },
                "standardize": True,
                "normalize": False,
                "initial_noise_val": 1.0,
                "noise_constraint": 1.0e-4,
                "initial_outputscale_val": 1.0,
                "initial_lengthscale_val": 1.0,
                "gp_lr": 0.2,
                "ft_lr": 0.002,
                "scale_embeddings": True,
                "train_mll_additionally": False,
                "finetuning_model": {
                    "class_path": "gollum.featurization.deep.LLMFeaturizer",
                    "init_args": {
                        "projection_dim": 64,
                        "from_top": True,
                        "target_ratio": 0.25,
                        "model_name": "t5-base",
                        "pooling_method": "average",
                        "normalize_embeddings": False,
                        "lora_dropout": 0.2,
                        "modules_to_save": None,
                        "trainable": True,
                        "input_dim": 768,
                    },
                },
            },
        },
        "n_iters": n_iters,
        "seed": seed,
    }


def setup_data(config: dict):
    initializer = instantiate_class(config["data"]["init_args"]["initializer"], seed=config["seed"])
    featurizer = instantiate_class(config["data"]["init_args"]["featurizer"])
    return instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=config["data"]["init_args"]["normalize_input"],
        maximize=config["data"]["init_args"]["maximize"],
    )


def setup_bo(config: dict, design_space: torch.Tensor):
    return BotorchOptimizer(
        design_space=design_space,
        surrogate_model_config=config["surrogate_model"],
        acq_function_config=config["acquisition"],
        batch_strategy=config["bo"]["init_args"]["batch_strategy"],
        batch_size=config["bo"]["init_args"]["batch_size"],
    )


def run_trial(data_path: Path, seed: int, total_budget: int, init_size: int) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("GOLLuM DeepGP baseline requires CUDA in this codebase.")

    bo_steps = total_budget - init_size
    config = build_config(data_path, seed, init_size, bo_steps)
    seed_everything(seed, workers=True)
    log(
        f"[GOLLuM] Trial {seed + 1} starting with init_size={init_size}, total_budget={total_budget}, "
        f"bo_steps={bo_steps}"
    )
    run = wandb.init(project="gollum-baseline", mode="disabled", reinit=True)
    try:
        dm = setup_data(config)
        dm.train_indexes = to_1d_int_numpy(dm.train_indexes)
        dm.heldout_indices = to_1d_int_numpy(dm.heldout_indices)
        dm.train_x = dm.x[dm.train_indexes]
        dm.train_y = dm.y[dm.train_indexes]
        dm.heldout_x = dm.x[dm.heldout_indices]
        dm.heldout_y = dm.y[dm.heldout_indices]
        bo = setup_bo(config, dm.heldout_x)
        trace = [float(dm.train_y.max().item())]
        log(
            f"[GOLLuM] Trial {seed + 1} initialized with {len(dm.train_indexes)} points, "
            f"remaining={len(dm.heldout_indices)}, best={trace[-1]:.4f}"
        )

        for step in range(bo_steps):
            log(
                f"[GOLLuM] Trial {seed + 1} iteration {step + 1}/{bo_steps}: "
                f"evaluated={len(dm.train_indexes)}, remaining={len(dm.heldout_indices)}, current_best={trace[-1]:.4f}"
            )
            train_x = dm.train_x.clone().to("cuda")
            train_y = dm.train_y.clone().to("cuda")
            design_space = dm.heldout_x.clone().to("cuda")
            stage_start = time.perf_counter()
            log(f"[GOLLuM] Trial {seed + 1} iteration {step + 1}: fitting DeepGP and optimizing acquisition...")
            x_next, indices = bo.suggest_next_experiments(
                train_x, train_y, design_space, return_indices=True
            )
            stage_elapsed = time.perf_counter() - stage_start
            log(f"[GOLLuM] Trial {seed + 1} iteration {step + 1}: model fit + acquisition finished in {stage_elapsed:.1f}s")
            indices_np = to_1d_int_numpy(indices)
            if indices_np.size != 1:
                raise RuntimeError(
                    f"Expected exactly one selected candidate, got {indices_np.size}. "
                    f"indices={indices_np.tolist()}"
                )
            chosen_original_indices_np = to_1d_int_numpy(dm.heldout_indices[indices_np])
            heldout_indices_np = to_1d_int_numpy(dm.heldout_indices)
            chosen_value = float(dm.y[chosen_original_indices_np[0]].item())
            dm.train_indexes = np.append(to_1d_int_numpy(dm.train_indexes), chosen_original_indices_np)
            dm.heldout_indices = np.delete(heldout_indices_np, indices_np)
            dm.train_indexes = to_1d_int_numpy(dm.train_indexes)
            dm.heldout_indices = to_1d_int_numpy(dm.heldout_indices)
            dm.train_x = dm.x[dm.train_indexes]
            dm.train_y = dm.y[dm.train_indexes]
            dm.heldout_x = dm.x[dm.heldout_indices]
            dm.heldout_y = dm.y[dm.heldout_indices]
            trace.append(float(dm.train_y.max().item()))
            log(
                f"[GOLLuM] Trial {seed + 1} selected idx={int(chosen_original_indices_np[0])} "
                f"observed={chosen_value:.4f} new_best={trace[-1]:.4f}"
            )
    finally:
        run.finish()

    log(f"[GOLLuM] Trial {seed + 1} finished with final_best={trace[-1]:.4f}")
    return np.array(trace, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Run GOLLuM DeepGP on DAR or OCM.")
    parser.add_argument("--dataset", choices=["dar", "ocm"], required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial-start-index", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--init-size", type=int, default=None)
    parser.add_argument("--append-to-existing", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cap_cpu_threads()
    hf_home = configure_runtime_paths()
    ensure_t5_available(hf_home)
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

    output_dir = Path(args.output_dir or ROOT / "outputs" / "baseline_runs" / "gollum" / args.dataset)
    results_path = output_dir / f"{args.dataset}_gollum_results.npz"
    requested_trial_numbers = np.arange(
        args.trial_start_index,
        args.trial_start_index + args.trials,
        dtype=int,
    )
    existing_payload = load_existing_results(results_path) if args.append_to_existing else None
    if existing_payload is None:
        existing_results = None
        existing_trial_numbers = np.empty(0, dtype=int)
    else:
        existing_results, existing_trial_numbers = existing_payload
        overlap = np.intersect1d(existing_trial_numbers, requested_trial_numbers)
        if overlap.size:
            overlap_text = ", ".join(f"trial_{trial_number:02d}" for trial_number in overlap.tolist())
            raise ValueError(f"Refusing to overwrite existing GOLLuM trial(s): {overlap_text}")
    data_path = build_processed_dataset(args.dataset, output_dir / "prepared_data")
    _, raw_df = load_benchmark_spec(ROOT, args.dataset)
    if args.total_budget > len(raw_df):
        raise ValueError(f"--total-budget={args.total_budget} exceeds dataset size {len(raw_df)}.")
    log(
        f"[GOLLuM][{args.dataset.upper()}] Loaded dataset with {len(raw_df)} rows, "
        f"budget={args.total_budget}, init_size={resolved_init_size}, trials={args.trials}, "
        f"trial_start_index={args.trial_start_index}, seed_start={args.seed_start}"
    )
    log(f"[GOLLuM][{args.dataset.upper()}] Writing outputs to {output_dir}")

    traces = []
    for offset in range(args.trials):
        seed = args.seed_start + offset
        traces.append(run_trial(data_path, seed, args.total_budget, resolved_init_size))

    new_results = np.vstack(traces)
    if existing_results is not None:
        results = np.vstack([existing_results, new_results])
        trial_numbers = np.concatenate([existing_trial_numbers, requested_trial_numbers])
    else:
        results = new_results
        trial_numbers = requested_trial_numbers
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(results_path, results=results, trial_numbers=trial_numbers)
    summary = {
        "dataset": args.dataset,
        "trials": int(len(trial_numbers)),
        "trial_numbers": trial_numbers.tolist(),
        "total_budget": args.total_budget,
        "init_size": resolved_init_size,
        "backbone_model": "t5-base",
        "initial_mean": float(results[:, 0].mean()),
        "final_mean": float(results[:, -1].mean()),
        "final_std": float(results[:, -1].std()),
    }
    (output_dir / f"{args.dataset}_gollum_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    log(
        f"[GOLLuM][{args.dataset.upper()}] Completed all trials. "
        f"initial_mean={summary['initial_mean']:.4f}, final_mean={summary['final_mean']:.4f}, final_std={summary['final_std']:.4f}"
    )


if __name__ == "__main__":
    main()
