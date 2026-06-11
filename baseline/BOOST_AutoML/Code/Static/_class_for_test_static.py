
import os
import gc
import random

import numpy as np
import torch
from tqdm import tqdm

from benchmarks.Benchmark_ftn import Benchmarks
from core.BOOST import BOOST
from core.BayesianOptimization import BayesianOptimizer
from core.kernels_and_acquisitions import KernelType, AcquisitionType
from utils.Save_results import save_final_data_to_excel

os.environ['OMP_NUM_THREADS'] = '1'

class TestFunction(BayesianOptimizer):
    def __init__(
            self,
            device='cpu',
            use_xue=False,
            kernel_type=KernelType.TBD,
            acquisition_type=AcquisitionType.TBD,
            objective=Benchmarks.Ackley,
            bounds=None,
            n_grid=21,
            dim=4,
            target=1e-2,
            max_iter=100,
            n_init_points=10,
            seed=0,
            base_dir=None,
            is_fixed_candidate_x=False,
            candidate_x=None,
            candidate_y=None,
            ):
        super().__init__(device=device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_type = kernel_type
        self.acquisition_type = acquisition_type
        self.use_xue = use_xue
        self.objective = objective
        self.bounds = bounds
        self.n_grid = n_grid
        self.dim = dim
        self.target = target

        self.max_iter= max_iter
        self.n_init_points = n_init_points
        self.seed = seed
        self.base_dir = base_dir

        self.is_fixed_candidate_x = is_fixed_candidate_x
        if candidate_x is not None:
            self.candidate_x = candidate_x.to(self.device)
        else:
            self.candidate_x = None
        if candidate_y is not None:
            self.candidate_y = candidate_y.to(self.device)
        else:
            self.candidate_y = None

        self.all_indices = None
        self.train_indices = None
        self.filtered_candidate_x = None
        self.filtered_candidate_y = None

    def optimize_recommend_adaptive(self):
        torch.set_default_dtype(torch.double)
        self.set_seed(self.seed)
        if self.is_fixed_candidate_x:
            # Generate training set
            num_data = self.candidate_x.shape[0]
            index_initial_sample = np.random.choice(num_data, self.n_init_points, replace=False)
            self.train_x = self.candidate_x[index_initial_sample]
            self.train_y = self.candidate_y[index_initial_sample]

            mask = torch.ones(self.candidate_x.shape[0], dtype=torch.bool, device=self.device)
            mask[torch.tensor(index_initial_sample, device=self.device)] = False
            self.filtered_candidate_x = self.candidate_x[mask]
            self.filtered_candidate_y = self.candidate_y[mask]

            self.target = self.candidate_y.min()

        else:
            self.train_x = self._generate_lhs_samples(dim=self.dim, n_samples=self.n_init_points, bounds=self.bounds, n_grid=self.n_grid).to(self.device)
            self.train_y = self.objective(self.train_x).to(dtype=self.train_x.dtype).to(self.device)

            candidate_points = []
            for d in range(self.dim):
                candidate_points.append(torch.linspace(self.bounds[0], self.bounds[1], self.n_grid, device=self.device))
            self.candidate_x = torch.cartesian_prod(*candidate_points).to(self.device)

            # Remove already selected points from candidate_x
            mask = ~torch.any(torch.cdist(self.candidate_x, self.train_x) < 1e-5, dim=1)
            self.filtered_candidate_x = self.candidate_x[mask]

        current_min = self.train_y.min().item()


        # Initialize progress bar
        bar_format = '{desc}: {percentage:3.0f}%|{bar:10}| {n:3d}/{total:3d} [{elapsed}<{remaining}, {rate_fmt}]{postfix}'
        desc = f"{self.kernel_type.value:>8}_{self.acquisition_type.value:>6}_{self.seed + 1:2d}"
        pbar = tqdm(range(self.n_init_points, self.max_iter), desc=desc, bar_format=bar_format)



        # Generate dictionary to save results
        history = {
            'iterations': [],
            'best_values': [],
        }
        for i in range(self.n_init_points):
            history['iterations'].append(i)
            regret = self.train_y[:i+1].min().item()-self.target
            value = regret.item() if isinstance(regret, torch.Tensor) else regret
            history['best_values'].append(value)

        for iter in pbar:
            # Use BOOST to get recommendation of kernel and acquisition functions
            if self.use_xue and iter == 10:
                boost = BOOST(device=self.device)
                self.kernel_type, self.acquisition_type = boost.get_kernel_acq(train_x=self.train_x, train_y=self.train_y, objective=self.objective, iter=iter, seed=self.seed, n_init_points=self.n_init_points, base_dir=self.base_dir)
            # reset seed to be dependent of seed in BOOST
            self.set_seed(self.seed)

            # Get next point using BO
            next_x, next_y, next_x_idx = self.get_next_point(train_x=self.train_x, train_y=self.train_y, filtered_candidate_x=self.filtered_candidate_x, filtered_candidate_y=self.filtered_candidate_y, kernel_type=self.kernel_type, acquisition_type=self.acquisition_type, objective=self.objective)
            # update train_x and train_y
            self.train_x = torch.cat([self.train_x, next_x], dim=0)
            self.train_y = torch.cat([self.train_y, next_y], dim=0)

            mask = torch.ones(self.filtered_candidate_x.shape[0], dtype=torch.bool, device=self.device)
            mask[next_x_idx] = False
            self.filtered_candidate_x = self.filtered_candidate_x[mask]
            assert (self.filtered_candidate_x.shape[0] + self.train_x.shape[0] - self.candidate_x.shape[0]) == 0
            if self.is_fixed_candidate_x:
                self.filtered_candidate_y = self.filtered_candidate_y[mask]

            # update current best
            current_min = self.train_y.min().item()
            best_idx = self.train_y.argmin().item()
            best_x = self.train_x[best_idx]

            # Update progress bar
            desc = f"{self.kernel_type.value:>8}_{self.acquisition_type.value:>6}_{self.seed + 1:2d}"
            pbar.set_description(desc)
            postfix = (
                f"Best ={current_min-self.target:>8.3f}, "
                f"Best pos = [{', '.join(f'{x:>7.3f}' for x in best_x.cpu().numpy())}], "
                f"Current pos = [{', '.join(f'{x:>7.3f}' for x in next_x[0].cpu().numpy())}]"
            )
            pbar.set_postfix_str(postfix)

            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # update history
            history['iterations'].append(iter)
            regret = current_min - self.target
            value = regret.item() if isinstance(regret, torch.Tensor) else regret
            history['best_values'].append(value)

            # stop if target reached
            if self.train_y.min().item() <= self.target + 1e-10:
                remaining_iterations = range(iter + 1, self.max_iter)
                for remaining_iter in remaining_iterations:
                    history['iterations'].append(remaining_iter)
                    history['best_values'].append(0.0)
                break

        pbar.close()

        save_final_data_to_excel(
            self.train_x, self.train_y, self.seed, self.kernel_type, self.acquisition_type, self.objective, self.base_dir
        )

        return {
            'kernel': self.kernel_type,
            'acquisition': self.acquisition_type,
            'seed': self.seed,
            'final_best': current_min,
            'iterations': history['iterations'],
            'best_values': history['best_values'],
        }


    def _generate_lhs_samples(self, dim, n_samples, bounds=None, n_grid=19):
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

        return torch.tensor(list(generated_samples)[:n_samples], dtype=torch.double, device=self.device)


