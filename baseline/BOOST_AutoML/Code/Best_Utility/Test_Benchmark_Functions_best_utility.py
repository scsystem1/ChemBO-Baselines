from pathlib import Path
import os
import sys


_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import time
from datetime import datetime

from _class_for_test_best_utility import TestFunction
from benchmarks.Benchmark_ftn import Benchmarks
from core.kernels_and_acquisitions import KernelType, AcquisitionType
from utils.Save_results import save_individual_trial, save_kernel_log_to_excel

def cleanup_resources(full=False): # clean resources if needed
    import gc, os, torch, multiprocessing as mp
    import shutil
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    safe_delete("joblib_memmapping_folder_", "/tmp")
    if full:
        safe_delete("joblib", "/dev/shm")
        safe_delete("loky", "/dev/shm")

def test_benchmark_function(use_boost=False, kernels=[KernelType.TBD], acquisitions=[AcquisitionType.TBD], benchmarks=None, n_init_points=10, max_iter=100, trial=10):
    base_dir = str(_PROJECT_DIR / f'Best_Utility/results_BestUtility_{datetime.now().strftime("%Y%m%d")}')
    os.makedirs(base_dir, exist_ok=True)

    for objective, config in benchmarks:
        print(f"\nTesting {objective.__name__} function")
        time.sleep(0.5)
        for acquisition_type in acquisitions:
            for kernel_type in kernels:
                current_trial_results = []
                for i in range(trial):
                    time.sleep(0.5)
                    test = TestFunction(
                        device='cpu',
                        use_boost=use_boost,
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

                    result, kernel_log = test.optimize_recommend_adaptive()

                    save_kernel_log_to_excel(
                        log_data=kernel_log,
                        seed=i,
                        objective=objective.__name__,
                        base_dir=base_dir
                    )

                    current_trial_results.append({
                        'objective': objective.__name__,
                        'seed': i,
                        'method': 'recommended' if use_boost else 'determined',
                        **result
                    })
                    save_individual_trial(current_trial_results, objective.__name__, n_initial_points=n_init_points, base_dir=base_dir)

                    time.sleep(0.5)
                    cleanup_resources()
                save_individual_trial(current_trial_results, objective.__name__, n_initial_points=n_init_points, base_dir=base_dir)
                cleanup_resources(full=True)


if __name__ == '__main__':
    cleanup_resources(full=True)
    benchmarks = [
        (Benchmarks.Ackley, Benchmarks.ACKLEY_CONFIG),
        (Benchmarks.Levy, Benchmarks.LEVY_CONFIG),
        (Benchmarks.Rosenbrock, Benchmarks.ROSENBROCK_CONFIG),
    ]
    kernels = [KernelType.TBD]
    acquisitions = [AcquisitionType.EI]
    test_benchmark_function(use_boost=False, kernels=kernels, acquisitions=acquisitions, benchmarks=benchmarks, n_init_points=10, max_iter=100, trial=30)