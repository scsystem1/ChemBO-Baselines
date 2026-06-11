import os
import sys


_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import GPyOpt
import GPy
from GPyOpt.core.task.space import Design_space
import torch
import pandas as pd
import random
from enum import Enum
from pathlib import Path


from nopast import get_best_evaluation

from datetime import datetime
from utils.Save_results import save_final_data_to_excel, save_acquisition_log_to_excel, save_individual_trial


class KernelType(Enum):
    MATERN52 = "Matern52"


class AcquisitionType(Enum):
    Hedge = "Hedge"


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

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


acquisitions = [{'type': 'ei', 'epsilon': 0}, {'type': 'pi', 'epsilon': 0}, {'type': 'lcb', 'upsilon': 0.1, 'delta': 0},
                {'type': 'mean'}, ]

objective_function = [
        [2277, 15],
        [5636, 6],
        [5891, 8],
        [5906, 16],
        [5964, 9],
        [6322, 8],
        [6762, 6],
        [6794, 10],
]  # space number, dimension

target = 0

base_dir = str(_PROJECT_DIR / f'Adaptive_Acquisition/results_HPOB_Hedge_{datetime.now().strftime("%Y%m%d")}')
os.makedirs(base_dir, exist_ok=True)

for data_num, dim in objective_function:
    objective_name = f'{data_num}_{dim}D'
    print(f"\nTesting {objective_name} function")
    current_trial_results = []
    candidate_x, candidate_y = HPOB(objective=objective_name).get_data()
    dim = dim if dim is not None else candidate_x.shape[1]
    candidate_x = candidate_x.numpy()
    candidate_y = candidate_y.numpy()

    def ftn(x):
        idx = np.where((candidate_x == x).all(axis=1))[0][0]
        return candidate_y[idx]
    space = Design_space([
        {'name': 'config', 'type': 'bandit', 'domain': candidate_x}
    ])

    for i in range(30):
        print(f"\nStarting optimization for {data_num}_{dim}D, Trial {i + 1}/30")
        set_seed(i)
        index_initial_sample = np.random.choice(candidate_x.shape[0], 10, replace=False)
        X_init = candidate_x[index_initial_sample]
        y_init = np.array([ftn(x) for x in X_init]).reshape(-1, 1)
        set_seed(i)
        evaluations, scores, acquisition_log = get_best_evaluation(X_init, y_init, space, acquisitions, ftn, factor=0.7,
                                                                   iterations=90, eta=4, target=target)

        # Save Results
        train_x = torch.from_numpy(evaluations[:, :-1])
        train_y = torch.from_numpy(evaluations[:, -1])

        max_total_iterations = 10 + 90  # n_samples + iterations
        best_values_so_far = np.minimum.accumulate(train_y.numpy()).tolist()

        if len(best_values_so_far) < max_total_iterations:
            padding_value = target
            padding_length = max_total_iterations - len(best_values_so_far)
            best_values_so_far.extend([padding_value] * padding_length)

        history = {
            'iterations': list(range(max_total_iterations)),
            'best_values': best_values_so_far
        }
        current_trial_results.append({
            'objective': f'{data_num}_{dim}D',
            'seed': i,
            'method': 'Hedge',
            'final_best': train_y.min().item(),
            'kernel': KernelType.MATERN52,
            'acquisition': AcquisitionType.Hedge,
            **history
        })

        save_final_data_to_excel(
            train_x=train_x,
            train_y=train_y,
            seed=i,
            kernel_type=KernelType.MATERN52,
            acquisition_type=AcquisitionType.Hedge,
            objective=f'{data_num}_{dim}D',
            base_dir=base_dir
        )

        save_acquisition_log_to_excel(
            log_data=acquisition_log,
            acquisition_names=[acq['type'] for acq in acquisitions],
            seed=i,
            objective=f'{data_num}_{dim}D',
            base_dir=base_dir
        )

        save_individual_trial(current_trial_results, f'{data_num}_{dim}D', n_initial_points=10, base_dir=base_dir)
