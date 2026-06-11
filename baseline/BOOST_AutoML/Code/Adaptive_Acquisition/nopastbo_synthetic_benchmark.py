from pathlib import Path
import os
import sys

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import GPyOpt
import GPy
from GPyOpt.core.task.space import Design_space
import torch
import random
from enum import Enum

from nopast import get_best_evaluation
from nopastbo_benchmarks import ackley, levy, rosenbrock

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

def _generate_lhs_samples(dim, n_samples, bounds=None, n_grid=19):
    """LHS for discrete grid points."""

    # To check already evaluated points
    generated_samples = set()

    # In cases where n_samples is larger than the number of 1D grid points,
    # we repeat the LHS procedure multiple times to generate sufficient samples.
    # In this study, aside from this adjustment for large n_samples,
    # the sampling method is identical to the standard LHS.
    while len(generated_samples) < n_samples:
    # To check already evaluated points
        lhs_points = []
        for d in range(dim):
            # discrete version of LHS
            grid_points = torch.linspace(bounds[0], bounds[1], n_grid)
            lhs_step = max(1, (n_grid - 1) // (n_samples - 1))
            lhs_start = ((n_grid - 1) - lhs_step * (n_samples - 1)) // 2
            dim_points = [grid_points[lhs_start + i * lhs_step].item() for i in range(min(n_samples, n_grid))]
            random.shuffle(dim_points)
            lhs_points.append(dim_points)

        # add to generated_samples
        new_points = list(zip(*lhs_points))
        for point in new_points:
            generated_samples.add(point)
            if len(generated_samples) >= n_samples:
                break

    return np.array(list(generated_samples)[:n_samples])

acquisitions = [{'type': 'ei','epsilon': 0},{'type': 'pi','epsilon': 0},{'type': 'lcb','upsilon': 0.1,'delta': 0},{'type': 'mean'},]

objective_function = [
    [ackley, "Ackley", -31.5, 31.5, 37, 0],       # f(0,0,0,0) = 0
    [levy, "Levy", -10.0, 10.0, 41, 0],           # f(1,1,1,1) = 0
    [rosenbrock, "Rosenbrock", -5.0, 10.0, 31, 0],# f(1,1,1,1) = 0
] # ftn, name, lb, ub, n_grid, target

base_dir = str(_PROJECT_DIR / f'Adaptive_Acquisition/results_Hedge_{datetime.now().strftime("%Y%m%d")}')
os.makedirs(base_dir, exist_ok=True)


for ftn, name, lb, ub, n_grid, target in objective_function:
    current_trial_results = []
    discrete_values = tuple(np.round(np.linspace(lb, ub, n_grid), 8))
    space = Design_space([
        {'name': 'x1', 'type': 'discrete', 'domain': discrete_values},
        {'name': 'x2', 'type': 'discrete', 'domain': discrete_values},
        {'name': 'x3', 'type': 'discrete', 'domain': discrete_values},
        {'name': 'x4', 'type': 'discrete', 'domain': discrete_values}
    ])
    for i in range(1):
        print(f"\nStarting optimization for {name}, Trial {i+1}/30")
        set_seed(i)
        X_init = _generate_lhs_samples(dim=4, n_samples=10, bounds=[lb,ub], n_grid=n_grid)
        y_init = ftn(X_init)
        set_seed(i)
        evaluations, scores, acquisition_log = get_best_evaluation(X_init, y_init, space, acquisitions, ftn, factor=0.7, iterations=90, eta=4, target=target)

        # Save Results
        train_x = torch.from_numpy(evaluations[:, :-1])
        train_y = torch.from_numpy(evaluations[:, -1])

        max_total_iterations = 10 + 90 # n_samples + iterations
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
            'objective': name,
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
            objective=ftn.__name__,
            base_dir=base_dir
        )

        save_acquisition_log_to_excel(
            log_data=acquisition_log,
            acquisition_names=[acq['type'] for acq in acquisitions],
            seed=i,
            objective=ftn.__name__,
            base_dir=base_dir
        )

        save_individual_trial(current_trial_results, name, n_initial_points=10, base_dir=base_dir)
