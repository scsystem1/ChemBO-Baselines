import os
import sys


_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from _class_for_test_best_utility import TestFunction
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


class HPOB:
    def __init__(self, data_dir=None, objective=None):
        self.data_dir = Path(data_dir) if data_dir is not None else _PROJECT_DIR / 'benchmarks' / 'hpob_data'
        self.objective = objective

    def get_data(self):
        file_path = self.data_dir / f'{self.objective}.csv'
        maximize = False
        if os.path.exists(file_path):
            print(f"✅{self.objective}.csv exists")
        else:
            raise ValueError(f"❌❌❌File does not exist: {file_path}❌❌❌")

        df = pd.read_csv(file_path)
        candidate_x = torch.tensor(df.iloc[:, :-1].values)
        candidate_y = torch.tensor(df.iloc[:, -1].values)
        
        if maximize:
            candidate_y = -candidate_y  # maximize → minimize

        global_min = candidate_y.min()
        global_max = candidate_y.max()
        candidate_y = (candidate_y - global_min) / (global_max - global_min)

        return candidate_x, candidate_y

def test_hpob(use_boost=False, kernels=[KernelType.TBD], acquisitions=[AcquisitionType.TBD], benchmarks=None, n_init_points=10, max_iter=100, trial=10):
    base_dir = str(_PROJECT_DIR / f'Best_Utility/results_HPOB_BestUtility_{datetime.now().strftime("%Y%m%d")}')
    os.makedirs(base_dir, exist_ok=True)

    for data_num, dim in benchmarks:
        objective_name = f'{data_num}_{dim}D'
        print(f"\nTesting {objective_name} function")
        time.sleep(0.5)
        for acquisition_type in acquisitions:
            for kernel_type in kernels:
                current_trial_results = []
                for i in range(trial):
                    candidate_x, candidate_y = HPOB(objective=objective_name).get_data()
                    dim = dim if dim is not None else candidate_x.shape[1]
                    time.sleep(0.5)
                    test = TestFunction(
                        use_boost=use_boost,
                        kernel_type=kernel_type,
                        acquisition_type=acquisition_type,
                        objective=f'{data_num}_{dim}D',
                        dim=dim,
                        max_iter=max_iter,
                        n_init_points=n_init_points,
                        seed=i,
                        base_dir=base_dir,

                        is_fixed_candidate_x=True,
                        candidate_x=candidate_x,
                        candidate_y=candidate_y,
                    )

                    result, kernel_log = test.optimize_recommend_adaptive()

                    save_kernel_log_to_excel(
                        log_data=kernel_log,
                        seed=i,
                        objective=f'{data_num}_{dim}D',
                        base_dir=base_dir
                    )

                    current_trial_results.append({
                        'objective': f'{data_num}_{dim}D',
                        'seed': i,
                        'method': 'recommended' if use_boost else None,
                        **result
                    })
                    save_individual_trial(current_trial_results, f'{data_num}_{dim}D', n_initial_points=n_init_points, base_dir=base_dir)
                    cleanup_resources()

                    time.sleep(0.5)
                save_individual_trial(current_trial_results, f'{data_num}_{dim}D', n_initial_points=n_init_points, base_dir=base_dir)
                cleanup_resources(full=True)


if __name__ == '__main__':
    benchmark_list = [
        [2277, 15],
        [5636, 6],
        [5891, 8],
        [5906, 16],
        [5964, 9],
        [6322, 8],
        [6762, 6],
        [6794, 10],
    ]
    for benchmark in benchmark_list:
        cleanup_resources(full=True)
        benchmarks = [benchmark]

        kernels = [KernelType.TBD]
        acquisitions = [AcquisitionType.EI]
        test_hpob(use_boost=False, kernels=kernels, acquisitions=acquisitions, benchmarks=benchmarks, n_init_points=10, max_iter=100, trial=30)