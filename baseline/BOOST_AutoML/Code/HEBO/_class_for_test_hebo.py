"""HEBO baseline with KD-tree nearest-neighbour snap to a discrete candidate set.

HEBO is included as a strong external black-box optimization baseline. It
operates in continuous space, so to apply it to BOOST's discrete protocol
(synthetic 4-D grid + HPO-B fixed candidate sets) we snap each HEBO suggestion
to its nearest still-available point in the candidate set via scipy.spatial.cKDTree.

Design notes
------------
* Self-contained: this module deliberately does NOT import core.BayesianOptimization
  or core.kernels_and_acquisitions because the boost-hebo conda env lacks
  gpytorch / botorch (HEBO conflicts on those). Lightweight local enums supply
  the .value interface that Save_results.py expects.
* External initial points: HEBO's built-in random initialisation is bypassed
  (rand_sample=0). The 10 LHS initial points are generated with the SAME logic
  as Random_Search / BoTorch_Default baselines so per-seed comparisons are
  fair across methods.
* KD-tree snap: the tree is built once on the full candidate set; we mask the
  picked indices and re-query with k = observed_count + 1 so the closest still-
  available point is always returned without rebuilding the tree.
* Minimisation: HEBO minimises by default; train_y is fed as-is.
"""

import gc
import math
import os
import random
import warnings
from enum import Enum

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils.Save_results import save_final_data_to_excel

# HEBO emits noisy stderr from pymoo and torch deprecations under the boost-hebo stack.
warnings.filterwarnings("ignore")

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

os.environ["OMP_NUM_THREADS"] = "1"


class KernelType(Enum):
    """Lightweight stand-in so Save_results.py file naming stays consistent."""
    TBD = "TBD"
    HEBO = "HEBO"


class AcquisitionType(Enum):
    """HEBO uses MACE (Multi-objective ACquisition Ensemble) internally."""
    TBD = "TBD"
    MACE = "MACE"


class TestFunction:
    """HEBO BO loop with KD-tree snap to discrete candidates.

    Mirrors the constructor signature of Random_Search / BoTorch_Default
    TestFunction so the entry scripts look identical across baselines.
    """

    def __init__(
        self,
        device="cpu",
        kernel_type=KernelType.HEBO,
        acquisition_type=AcquisitionType.MACE,
        objective=None,
        bounds=None,
        n_grid=21,
        dim=4,
        target=0.0,
        max_iter=100,
        n_init_points=10,
        seed=0,
        base_dir=None,
        is_fixed_candidate_x=False,
        candidate_x=None,
        candidate_y=None,
    ):
        self.device = torch.device("cpu")  # HEBO is numpy/pandas based; CPU is correct.
        self.kernel_type = kernel_type
        self.acquisition_type = acquisition_type
        self.objective = objective
        self.bounds = bounds
        self.n_grid = n_grid
        self.dim = dim
        self.target = target

        self.max_iter = max_iter
        self.n_init_points = n_init_points
        self.seed = seed
        self.base_dir = base_dir

        self.is_fixed_candidate_x = is_fixed_candidate_x
        if candidate_x is not None:
            self.candidate_x = candidate_x.to(self.device).to(dtype=torch.double)
        else:
            self.candidate_x = None
        if candidate_y is not None:
            self.candidate_y = candidate_y.to(self.device).to(dtype=torch.double)
        else:
            self.candidate_y = None

        self.train_x = None
        self.train_y = None
        self.filtered_candidate_x = None
        self.filtered_candidate_y = None

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # use_deterministic_algorithms is NOT called here because HEBO/torch in this env
        # may use ops without deterministic implementations; np/random/torch seeds suffice.

    # ------------------------------------------------------------------
    # LHS — verbatim copy of the Random_Search / BoTorch_Default routine
    # so per-seed initial sets are byte-identical across baselines.
    # ------------------------------------------------------------------
    def _generate_lhs_samples(self, dim, n_samples, bounds=None, n_grid=19):
        generated_samples = set()
        while len(generated_samples) < n_samples:
            lhs_points = []
            for _ in range(dim):
                grid_points = torch.linspace(bounds[0], bounds[1], n_grid)
                lhs_step = max(1, (n_grid - 1) // (n_samples - 1))
                lhs_start = ((n_grid - 1) - lhs_step * (n_samples - 1)) // 2
                dim_points = [grid_points[lhs_start + i * lhs_step].item()
                              for i in range(min(n_samples, n_grid))]
                random.shuffle(dim_points)
                lhs_points.append(dim_points)

            new_points = list(zip(*lhs_points))
            for point in new_points:
                generated_samples.add(point)
                if len(generated_samples) >= n_samples:
                    break

        return torch.tensor(list(generated_samples)[:n_samples], dtype=torch.double, device=self.device)

    # ------------------------------------------------------------------
    # Main BO loop
    # ------------------------------------------------------------------
    def optimize_recommend_adaptive(self):
        torch.set_default_dtype(torch.double)
        self.set_seed(self.seed)

        # ---- 1. Build candidate set + initial points -------------------
        if self.is_fixed_candidate_x:
            num_data = self.candidate_x.shape[0]
            index_initial_sample = np.random.choice(num_data, self.n_init_points, replace=False)
            self.train_x = self.candidate_x[index_initial_sample]
            self.train_y = self.candidate_y[index_initial_sample]

            self.target = self.candidate_y.min().item()
            full_candidate_np = self.candidate_x.cpu().numpy().astype(np.float64)
            full_candidate_y_np = self.candidate_y.cpu().numpy().astype(np.float64)

            init_indices = set(int(i) for i in index_initial_sample.tolist())
            # HEBO design space: each input dim is numerical in the observed [min, max] of the
            # candidate set so HEBO's internal scaler matches the data range.
            lb = full_candidate_np.min(axis=0)
            ub = full_candidate_np.max(axis=0)
            # Ensure lb < ub even for constant dims.
            ub = np.where(ub - lb < 1e-12, lb + 1e-6, ub)
            space_config = [
                {"name": f"x{i}", "type": "num", "lb": float(lb[i]), "ub": float(ub[i])}
                for i in range(self.dim)
            ]
        else:
            self.train_x = self._generate_lhs_samples(
                dim=self.dim, n_samples=self.n_init_points, bounds=self.bounds, n_grid=self.n_grid,
            ).to(self.device)
            self.train_y = self.objective(self.train_x).to(dtype=self.train_x.dtype).to(self.device)

            candidate_axes = [
                torch.linspace(self.bounds[0], self.bounds[1], self.n_grid, dtype=torch.double, device=self.device)
                for _ in range(self.dim)
            ]
            self.candidate_x = torch.cartesian_prod(*candidate_axes).to(self.device)
            full_candidate_np = self.candidate_x.cpu().numpy().astype(np.float64)
            full_candidate_y_np = None  # synthetic: query objective on demand

            # Map each initial LHS row to its index in the cartesian-product candidate grid
            # so the HEBO/snap bookkeeping stays consistent.
            init_indices = self._match_indices(self.train_x.cpu().numpy().astype(np.float64), full_candidate_np)
            init_indices = set(int(i) for i in init_indices)

            space_config = [
                {"name": f"x{i}", "type": "num", "lb": float(self.bounds[0]), "ub": float(self.bounds[1])}
                for i in range(self.dim)
            ]

        # ---- 2. KD-tree built once on the full candidate set -----------
        kdtree = cKDTree(full_candidate_np)
        available_mask = np.ones(full_candidate_np.shape[0], dtype=bool)
        for idx in init_indices:
            available_mask[idx] = False

        # ---- 3. HEBO optimiser with external initial observations ------
        space = DesignSpace().parse(space_config)
        # rand_sample=0 disables HEBO's internal random init so we control the warm-up set.
        # model_name='gp' is HEBO's default; MACE acquisition is the default.
        opt = HEBO(space, model_name="gp", rand_sample=0, scramble_seed=self.seed)

        col_names = [f"x{i}" for i in range(self.dim)]
        X_init_df = pd.DataFrame(self.train_x.cpu().numpy(), columns=col_names)
        y_init_arr = self.train_y.cpu().numpy().reshape(-1, 1).astype(np.float64)
        opt.observe(X_init_df, y_init_arr)

        # ---- 4. History bookkeeping -------------------------------------
        current_min = float(self.train_y.min().item())
        history = {"iterations": [], "best_values": []}
        for i in range(self.n_init_points):
            history["iterations"].append(i)
            history["best_values"].append(float(self.train_y[: i + 1].min().item()) - self.target)

        # ---- 5. BO loop -------------------------------------------------
        bar_format = "{desc}: {percentage:3.0f}%|{bar:10}| {n:3d}/{total:3d} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        desc = f"HEBO_seed{self.seed + 1:2d}"
        pbar = tqdm(range(self.n_init_points, self.max_iter), desc=desc, bar_format=bar_format)

        for it in pbar:
            # Reseed each iter so HEBO's internal NSGA-II / MC samplers stay deterministic
            # under (seed, iter), mirroring the BoTorch baseline's reseed pattern.
            self.set_seed(self.seed * 10000 + it)

            try:
                proposal_df = opt.suggest(n_suggestions=1)
            except Exception as e:
                # If HEBO's GP fit / acquisition optimisation fails (rare numerical issues),
                # fall back to a uniform-random unobserved candidate so the trial still produces
                # a complete history. This matches how BoTorch baselines log a fallback path.
                warnings.warn(f"HEBO suggest failed at iter {it} (seed {self.seed}): {e}. Using random fallback.")
                avail_idx = np.flatnonzero(available_mask)
                rand_choice = int(np.random.choice(avail_idx))
                proposal_df = pd.DataFrame(full_candidate_np[rand_choice].reshape(1, -1), columns=col_names)

            x_proposed = proposal_df[col_names].values.astype(np.float64)  # (1, dim)

            # KD-tree snap to the nearest still-available candidate.
            picked_idx = self._snap_to_available(kdtree, x_proposed[0], available_mask)
            next_x_np = full_candidate_np[picked_idx].reshape(1, -1)

            if self.is_fixed_candidate_x:
                next_y_val = float(full_candidate_y_np[picked_idx])
            else:
                next_x_torch = torch.tensor(next_x_np, dtype=torch.double, device=self.device)
                next_y_val = float(self.objective(next_x_torch).item())

            # Feed the snapped (x, y) back to HEBO so its surrogate is conditioned on the
            # value at the actually evaluated point, not the (continuous) proposal.
            X_obs_df = pd.DataFrame(next_x_np, columns=col_names)
            y_obs_arr = np.array([[next_y_val]], dtype=np.float64)
            opt.observe(X_obs_df, y_obs_arr)

            # Update bookkeeping
            available_mask[picked_idx] = False
            self.train_x = torch.cat(
                [self.train_x, torch.tensor(next_x_np, dtype=torch.double, device=self.device)], dim=0
            )
            self.train_y = torch.cat(
                [self.train_y, torch.tensor([next_y_val], dtype=torch.double, device=self.device)], dim=0
            )

            current_min = float(self.train_y.min().item())
            best_idx = int(self.train_y.argmin().item())
            best_x = self.train_x[best_idx].cpu().numpy()

            postfix = (
                f"Best ={current_min - self.target:>8.3f}, "
                f"Best pos = [{', '.join(f'{x:>7.3f}' for x in best_x)}], "
                f"Current pos = [{', '.join(f'{x:>7.3f}' for x in next_x_np[0])}]"
            )
            pbar.set_postfix_str(postfix)

            history["iterations"].append(it)
            history["best_values"].append(current_min - self.target)

            gc.collect()

            if current_min <= self.target + 1e-10:
                for remaining_iter in range(it + 1, self.max_iter):
                    history["iterations"].append(remaining_iter)
                    history["best_values"].append(0.0)
                break

        pbar.close()

        save_final_data_to_excel(
            self.train_x, self.train_y, self.seed,
            self.kernel_type, self.acquisition_type, self.objective, self.base_dir,
        )

        return {
            "kernel": self.kernel_type,
            "acquisition": self.acquisition_type,
            "seed": self.seed,
            "final_best": current_min,
            "iterations": history["iterations"],
            "best_values": history["best_values"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _match_indices(query_pts, candidate_set, tol=1e-5):
        """Return the candidate-set row index for each query point (synthetic LHS init).

        The LHS initial points lie ON the cartesian-product grid by construction, so
        a tight cdist threshold suffices.
        """
        diffs = np.sqrt(((query_pts[:, None, :] - candidate_set[None, :, :]) ** 2).sum(axis=-1))
        idxs = diffs.argmin(axis=1)
        # Sanity: every match must actually be within tolerance.
        for q, i in enumerate(idxs):
            if diffs[q, i] > tol:
                raise RuntimeError(
                    f"LHS row {q} did not match any candidate within tol={tol}; "
                    f"closest distance was {diffs[q, i]:.3e}."
                )
        return idxs

    @staticmethod
    def _snap_to_available(kdtree, x_query, available_mask):
        """Find the nearest still-available candidate index without rebuilding the tree.

        Strategy: query with growing k until we hit an unmasked index. With ~1.87M
        candidates and at most 90 picked points per trial, k stays tiny in practice
        (almost always 1).
        """
        n_total = available_mask.shape[0]
        k = 1
        while k <= n_total:
            k_eff = min(k, n_total)
            _, idxs = kdtree.query(x_query, k=k_eff)
            if k_eff == 1:
                idxs = np.array([idxs])
            for idx in np.atleast_1d(idxs):
                if available_mask[int(idx)]:
                    return int(idx)
            # No unmasked index in the top k; expand search.
            k = max(k * 2, k + 8)
        raise RuntimeError("No available candidates left in the candidate set.")
