"""HEBO baseline driver for the 3 synthetic benchmark functions (Ackley/Levy/Rosenbrock).

Run from the repository root in the boost-hebo conda env:

    conda run -n boost-hebo python Code/HEBO/Test_Benchmark_Functions_hebo.py

Outputs land in Results-style directories named
`results/results_HEBO_<YYYYMMDD>_trial30/` (relative to the cwd at launch).
"""

from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_PROJECT_DIR = Path(__file__).resolve().parent.parent

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
from datetime import datetime

from _class_for_test_hebo import TestFunction, KernelType, AcquisitionType
from benchmarks.Benchmark_ftn import Benchmarks
from utils.Save_results import save_individual_trial


def cleanup_resources(full=False):
    import gc, multiprocessing as mp, shutil

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
    for p in mp.active_children():
        try:
            p.terminate()
        except Exception:
            pass
    safe_delete("joblib_memmapping_folder_", "/tmp")
    if full:
        safe_delete("joblib", "/dev/shm")
        safe_delete("loky", "/dev/shm")


def test_benchmark_function(kernels=None, acquisitions=None, benchmarks=None,
                            n_init_points=10, max_iter=100, trial=30):
    if kernels is None:
        kernels = [KernelType.HEBO]
    if acquisitions is None:
        acquisitions = [AcquisitionType.MACE]

    base_dir = str(_PROJECT_DIR / f"results/results_HEBO_{datetime.now().strftime('%Y%m%d')}_trial{trial}")
    os.makedirs(base_dir, exist_ok=True)

    for objective, config in benchmarks:
        print(f"\nTesting {objective.__name__} function with HEBO baseline")
        time.sleep(0.5)
        for acquisition_type in acquisitions:
            for kernel_type in kernels:
                current_trial_results = []
                for i in range(trial):
                    time.sleep(0.5)
                    test = TestFunction(
                        device="cpu",
                        kernel_type=kernel_type,
                        acquisition_type=acquisition_type,
                        objective=objective,
                        bounds=config.bounds,
                        n_grid=config.n_grid,
                        dim=config.dim,
                        target=config.target,
                        max_iter=max_iter,
                        n_init_points=n_init_points,
                        seed=i,
                        base_dir=base_dir,
                    )

                    result = test.optimize_recommend_adaptive()

                    current_trial_results.append({
                        "objective": objective.__name__,
                        "seed": i,
                        "method": "determined",
                        **result,
                    })
                    save_individual_trial(current_trial_results, objective.__name__,
                                          n_initial_points=n_init_points, base_dir=base_dir)

                    time.sleep(0.5)
                    cleanup_resources()
                save_individual_trial(current_trial_results, objective.__name__,
                                      n_initial_points=n_init_points, base_dir=base_dir)
                cleanup_resources(full=True)


if __name__ == "__main__":
    cleanup_resources(full=True)
    benchmarks = [
        (Benchmarks.Ackley, Benchmarks.ACKLEY_CONFIG),
        (Benchmarks.Levy, Benchmarks.LEVY_CONFIG),
        (Benchmarks.Rosenbrock, Benchmarks.ROSENBROCK_CONFIG),
    ]

    test_benchmark_function(benchmarks=benchmarks, n_init_points=10, max_iter=100, trial=30)
