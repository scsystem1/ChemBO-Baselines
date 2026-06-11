import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"

IMPORT_PATHS = [
    CODE_DIR,
    CODE_DIR / "tests",
    CODE_DIR / "Static",
    CODE_DIR / "Random_Search",
    CODE_DIR / "Best_Utility",
    CODE_DIR / "Adaptive_Acquisition",
    CODE_DIR / "BoTorch_Default",
]

for import_path in IMPORT_PATHS:
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HPOB_TASKS = {
    "2277_15d": (2277, 15),
    "5636_6d": (5636, 6),
    "5891_8d": (5891, 8),
    "5906_16d": (5906, 16),
    "5964_9d": (5964, 9),
    "6322_8d": (6322, 8),
    "6762_6d": (6762, 6),
    "6794_10d": (6794, 10),
}

METHOD_CHOICES = ["boost", "fixed", "static", "random_search", "best_utility", "botorch_default", "boost_fixedhp", "random_kernel_af"]


def load_kernel_acquisition_types():
    from core.kernels_and_acquisitions import AcquisitionType, KernelType

    return KernelType, AcquisitionType


def load_synthetic_objectives():
    from benchmarks.Benchmark_ftn import Benchmarks

    return {
        "ackley": (Benchmarks.Ackley, Benchmarks.ACKLEY_CONFIG),
        "levy": (Benchmarks.Levy, Benchmarks.LEVY_CONFIG),
        "rosenbrock": (Benchmarks.Rosenbrock, Benchmarks.ROSENBROCK_CONFIG),
    }


def load_method_class(method):
    if method in {"boost", "fixed", "boost_fixedhp", "random_kernel_af"}:
        from _class_for_test_BOOST import TestFunction

        return TestFunction
    if method == "static":
        from _class_for_test_static import TestFunction

        return TestFunction
    if method == "random_search":
        from _class_for_test_randomsearch import TestFunction

        return TestFunction
    if method == "best_utility":
        from _class_for_test_best_utility import TestFunction

        return TestFunction
    if method == "botorch_default":
        from _class_for_test_botorch import TestFunction

        return TestFunction
    raise ValueError(f"Unsupported method: {method}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reviewer-friendly entry point for AutoML Appendix experiments."
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        required=False,
        help="Experiment method to run.",
    )
    parser.add_argument(
        "--space",
        choices=["synthetic", "hpob"],
        required=False,
        help="Search space family to run.",
    )
    parser.add_argument(
        "--objective",
        default="all",
        help="Synthetic objective name or comma-separated names: ackley, levy, rosenbrock, all.",
    )
    parser.add_argument(
        "--hpob-task",
        default="all",
        help="HPOB task id or comma-separated ids: 2277_15D, ..., 6794_10D, all.",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Kernel name: Matern32, Matern52, RBF, RQ, TBD.",
    )
    parser.add_argument(
        "--acquisition",
        default=None,
        help="Acquisition name: EI, PI, UCB, PM, TBD.",
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of random seeds to run.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Total optimization budget including initial points.",
    )
    parser.add_argument(
        "--n-init-points",
        type=int,
        default=10,
        help="Number of initial design points.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution device passed to the original classes.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where outputs should be written. Defaults to reviewer_runs/<timestamp>_<method>_<space>.",
    )
    parser.add_argument("--bounds-low", type=float, default=None, help="Synthetic bounds lower value override.")
    parser.add_argument("--bounds-high", type=float, default=None, help="Synthetic bounds upper value override.")
    parser.add_argument("--n-grid", type=int, default=None, help="Synthetic grid size override.")
    parser.add_argument("--dim", type=int, default=None, help="Synthetic dimension override.")
    parser.add_argument("--target", type=float, default=None, help="Synthetic target override.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available methods and tasks, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the resolved plan without running experiments or writing files.",
    )
    return parser.parse_args()


def print_catalog():
    synthetic_objectives = load_synthetic_objectives()
    print("Methods:")
    for method in METHOD_CHOICES:
        print(f"  - {method}")
    print("\nSynthetic objectives:")
    for name in synthetic_objectives:
        print(f"  - {name}")
    print("\nHPOB tasks:")
    for task in HPOB_TASKS:
        print(f"  - {task.upper()}")


def cleanup_resources(full=False):
    try:
        import torch
    except Exception:
        torch = None

    def safe_delete(prefix, path):
        if not os.path.exists(path):
            return
        for fname in os.listdir(path):
            if fname.startswith(prefix):
                fpath = os.path.join(path, fname)
                if os.path.isdir(fpath):
                    try:
                        shutil.rmtree(fpath)
                    except Exception:
                        pass

    gc.collect()
    for child in mp.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    safe_delete("joblib_memmapping_folder_", "/tmp")
    if full:
        safe_delete("joblib", "/dev/shm")
        safe_delete("loky", "/dev/shm")


def parse_kernel(value, method):
    KernelType, _ = load_kernel_acquisition_types()
    if value is None:
        if method in {"boost", "static", "random_search", "best_utility", "botorch_default", "boost_fixedhp", "random_kernel_af"}:
            return KernelType.TBD
        return None
    normalized = value.strip().upper()
    mapping = {
        "RBF": KernelType.RBF,
        "MATERN32": KernelType.MATERN32,
        "MATERN52": KernelType.MATERN52,
        "RQ": KernelType.RQ,
        "TBD": KernelType.TBD,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported kernel: {value}")
    return mapping[normalized]


def parse_acquisition(value, method):
    _, AcquisitionType = load_kernel_acquisition_types()
    if value is None:
        if method == "best_utility":
            return AcquisitionType.EI
        if method in {"boost", "static", "random_search", "botorch_default", "boost_fixedhp", "random_kernel_af"}:
            return AcquisitionType.TBD
        return None
    normalized = value.strip().upper()
    mapping = {
        "EI": AcquisitionType.EI,
        "PI": AcquisitionType.PI,
        "UCB": AcquisitionType.UCB,
        "PM": AcquisitionType.PM,
        "TBD": AcquisitionType.TBD,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported acquisition: {value}")
    return mapping[normalized]


def resolve_synthetic_configs(args):
    synthetic_objectives = load_synthetic_objectives()
    requested = [item.strip().lower() for item in args.objective.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        requested = list(synthetic_objectives.keys())

    resolved = []
    for name in requested:
        if name not in synthetic_objectives:
            raise ValueError(f"Unsupported synthetic objective: {name}")
        objective, config = synthetic_objectives[name]
        updated = replace(
            config,
            bounds=(
                args.bounds_low if args.bounds_low is not None else config.bounds[0],
                args.bounds_high if args.bounds_high is not None else config.bounds[1],
            ),
            n_grid=args.n_grid if args.n_grid is not None else config.n_grid,
            dim=args.dim if args.dim is not None else config.dim,
            target=args.target if args.target is not None else config.target,
        )
        resolved.append((objective, updated))
    return resolved


def resolve_hpob_tasks(args):
    requested = [item.strip().lower() for item in args.hpob_task.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        requested = list(HPOB_TASKS.keys())

    resolved = []
    for task in requested:
        if task not in HPOB_TASKS:
            raise ValueError(f"Unsupported HPOB task: {task}")
        resolved.append(HPOB_TASKS[task])
    return resolved


def validate_args(args):
    if args.list:
        return
    if args.method is None or args.space is None:
        raise ValueError("--method and --space are required unless --list is used.")
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.n_init_points <= 0:
        raise ValueError("--n-init-points must be positive.")
    if args.max_iter <= args.n_init_points:
        raise ValueError("--max-iter must be greater than --n-init-points.")
    if (args.bounds_low is None) ^ (args.bounds_high is None):
        raise ValueError("--bounds-low and --bounds-high must be provided together.")
    if args.method == "fixed":
        if args.kernel is None or args.acquisition is None:
            raise ValueError("--method fixed requires both --kernel and --acquisition.")


def get_output_dir(args):
    if args.output_dir:
        return ROOT / args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "reviewer_runs" / f"{timestamp}_{args.method}_{args.space}"


def load_hpob_data(objective_name):
    import pandas as pd
    import torch

    file_path = CODE_DIR / "benchmarks" / "hpob_data" / f"{objective_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"HPOB file not found: {file_path}")

    df = pd.read_csv(file_path)
    candidate_x = torch.tensor(df.iloc[:, :-1].values)
    candidate_y = torch.tensor(df.iloc[:, -1].values)

    global_min = candidate_y.min()
    global_max = candidate_y.max()
    candidate_y = (candidate_y - global_min) / (global_max - global_min)
    return candidate_x, candidate_y


def write_run_metadata(output_dir, args, run_items):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_config.json"
    payload = {
        "command": " ".join(sys.argv),
        "generated_at": datetime.now().isoformat(),
        "method": args.method,
        "space": args.space,
        "trials": args.trials,
        "max_iter": args.max_iter,
        "n_init_points": args.n_init_points,
        "device": args.device,
        "kernel": args.kernel,
        "acquisition": args.acquisition,
        "objectives": run_items,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_test_instance(method, args, objective, config=None, objective_name=None, candidate_x=None, candidate_y=None):
    TestFunction = load_method_class(method)
    kernel_type = parse_kernel(args.kernel, args.method)
    acquisition_type = parse_acquisition(args.acquisition, args.method)
    common_kwargs = {
        "device": args.device,
        "kernel_type": kernel_type,
        "acquisition_type": acquisition_type,
        "max_iter": args.max_iter,
        "n_init_points": args.n_init_points,
    }

    if args.space == "synthetic":
        common_kwargs.update(
            {
                "objective": objective,
                "bounds": config.bounds,
                "n_grid": config.n_grid,
                "dim": config.dim,
                "target": config.target,
            }
        )
    else:
        common_kwargs.update(
            {
                "objective": objective_name,
                "dim": candidate_x.shape[1],
                "is_fixed_candidate_x": True,
                "candidate_x": candidate_x,
                "candidate_y": candidate_y,
            }
        )

    if method == "boost":
        return TestFunction(use_boost=True, **common_kwargs)
    if method == "boost_fixedhp":
        return TestFunction(use_boost=True, fixed_hp_mode=True, **common_kwargs)
    if method == "random_kernel_af":
        return TestFunction(use_boost=True, random_select_mode=True, **common_kwargs)
    if method == "fixed":
        return TestFunction(use_boost=False, **common_kwargs)
    if method == "static":
        return TestFunction(use_xue=True, **common_kwargs)
    if method == "random_search":
        return TestFunction(**common_kwargs)
    if method == "best_utility":
        return TestFunction(use_boost=False, **common_kwargs)
    if method == "botorch_default":
        return TestFunction(**common_kwargs)
    raise ValueError(f"Unsupported method: {method}")


def run_single_problem(args, output_dir, label, objective=None, config=None, objective_name=None):
    from utils.Save_results import save_individual_trial, save_kernel_log_to_excel

    current_trial_results = []

    for seed in range(args.trials):
        if args.space == "hpob":
            candidate_x, candidate_y = load_hpob_data(objective_name)
        else:
            candidate_x = candidate_y = None

        test = make_test_instance(
            method=args.method,
            args=args,
            objective=objective,
            config=config,
            objective_name=objective_name,
            candidate_x=candidate_x,
            candidate_y=candidate_y,
        )
        test.seed = seed
        test.base_dir = str(output_dir)

        if args.method == "best_utility":
            result, kernel_log = test.optimize_recommend_adaptive()
            save_kernel_log_to_excel(
                log_data=kernel_log,
                seed=seed,
                objective=label,
                base_dir=str(output_dir),
            )
        else:
            result = test.optimize_recommend_adaptive()

        method_label = {
            "boost": "recommended",
            "boost_fixedhp": "recommended",
            "random_kernel_af": "recommended",
            "fixed": "determined",
            "static": "xue",
            "random_search": "determined",
            "best_utility": "determined",
            "botorch_default": "determined",
        }[args.method]

        current_trial_results.append(
            {
                "objective": label,
                "seed": seed,
                "method": method_label,
                **result,
            }
        )
        save_individual_trial(
            current_trial_results,
            label,
            n_initial_points=args.n_init_points,
            base_dir=str(output_dir),
        )
        cleanup_resources()

    save_individual_trial(
        current_trial_results,
        label,
        n_initial_points=args.n_init_points,
        base_dir=str(output_dir),
    )
    cleanup_resources(full=True)


def describe_plan(args):
    kernel = parse_kernel(args.kernel, args.method)
    acquisition = parse_acquisition(args.acquisition, args.method)
    output_dir = get_output_dir(args)
    if args.space == "synthetic":
        items = [objective.__name__ for objective, _ in resolve_synthetic_configs(args)]
    else:
        items = [f"{task_id}_{dim}D" for task_id, dim in resolve_hpob_tasks(args)]
    print("Resolved run plan")
    print(f"  method      : {args.method}")
    print(f"  space       : {args.space}")
    print(f"  items       : {', '.join(items)}")
    print(f"  trials      : {args.trials}")
    print(f"  max_iter    : {args.max_iter}")
    print(f"  n_init      : {args.n_init_points}")
    print(f"  kernel      : {kernel.value if kernel is not None else 'None'}")
    print(f"  acquisition : {acquisition.value if acquisition is not None else 'None'}")
    print(f"  output_dir  : {output_dir}")


def main():
    args = parse_args()

    if args.list:
        print_catalog()
        return

    validate_args(args)
    describe_plan(args)

    if args.dry_run:
        return

    output_dir = get_output_dir(args)
    if args.space == "synthetic":
        objectives = resolve_synthetic_configs(args)
        run_items = [objective.__name__ for objective, _ in objectives]
    else:
        tasks = resolve_hpob_tasks(args)
        run_items = [f"{task_id}_{dim}D" for task_id, dim in tasks]

    write_run_metadata(output_dir, args, run_items)

    if args.space == "synthetic":
        for objective, config in objectives:
            print(f"\nRunning {objective.__name__}")
            run_single_problem(
                args=args,
                output_dir=output_dir,
                label=objective.__name__,
                objective=objective,
                config=config,
            )
    else:
        for task_id, dim in tasks:
            objective_name = f"{task_id}_{dim}D"
            print(f"\nRunning {objective_name}")
            run_single_problem(
                args=args,
                output_dir=output_dir,
                label=objective_name,
                objective_name=objective_name,
            )

    print(f"\nCompleted. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
