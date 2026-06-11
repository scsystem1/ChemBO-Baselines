import gc
import os
import random

import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from core.BayesianOptimization import BayesianOptimizer
from core.kernels_and_acquisitions import KernelType, AcquisitionType
from utils.Save_results import save_recommendation_log

os.environ['OMP_NUM_THREADS'] = '1'


class BOOST(BayesianOptimizer):
    def __init__(
            self,
            is_fixed_candidate_x=True,
            kernel_candidates = [KernelType.MATERN32, KernelType.MATERN52, KernelType.RBF, KernelType.RQ],
            acquisition_candidates = [AcquisitionType.EI, AcquisitionType.PI, AcquisitionType.UCB, AcquisitionType.PM],
            device='cpu',
            fixed_hp_mode=False,
            fixed_hp_states=None,
            fixed_hp_normalize_stats=None,
            random_select_mode=False,
            trial_seed=None,
             ):
        super().__init__(device=device)
        self.is_fixed_candidate_x = is_fixed_candidate_x
        self.kernel_candidates = kernel_candidates
        self.acquisition_candidates = acquisition_candidates
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

        # Fixed-hyperparameter diagnostic path: when fixed_hp_mode=True, every internal
        # simulation step uses the kernel-specific hyperparameters cached on the external
        # full-data fit, instead of refitting on the reference set. Default False keeps
        # the original BOOST behaviour unchanged.
        self.fixed_hp_mode = fixed_hp_mode
        self.fixed_hp_states = fixed_hp_states  # dict[KernelType, {'model': sd, 'likelihood': sd}]
        self.fixed_hp_normalize_stats = fixed_hp_normalize_stats  # dict from external full-data normalize

        # Random-selection mode: when True, recommend() draws a (kernel, acquisition)
        # pair uniformly from the candidate pool and returns immediately, bypassing
        # the data partitioning and internal BO simulation. Mutually exclusive with
        # fixed_hp_mode. trial_seed is combined with the outer iteration index so
        # each (trial, iter) pair gets a deterministic but distinct draw.
        self.random_select_mode = random_select_mode
        self.trial_seed = trial_seed
        if random_select_mode and fixed_hp_mode:
            raise ValueError("random_select_mode and fixed_hp_mode are mutually exclusive.")

    def recommend(
            self,
            train_x_init=None,
            train_y_init=None,
            objective=None,
            seed=0,
            min_init_boost = 3,
            max_init_boost = 20,
            ratio_init_boost = 3, # Change this for different  |r_n| to |s_n| ratio
            max_iter_boost = 20,
    ):
        self.set_seed(seed)

        # Random-selection short-circuit: skip K-means + internal simulation entirely
        # and return a (kernel, acquisition) pair drawn from a local Random instance
        # seeded by (trial_seed, outer iter). Acquisition is drawn first to mirror the
        # default tie-breaking priority (acquisition outermost, kernel innermost).
        if self.random_select_mode:
            base_trial = self.trial_seed if self.trial_seed is not None else 0
            rng = random.Random(base_trial * 100003 + seed)
            chosen_acq = rng.choice(self.acquisition_candidates)
            chosen_kernel = rng.choice(self.kernel_candidates)
            return {
                'recommended_kernel': chosen_kernel.value,
                'recommended_acquisition': chosen_acq.value,
                'iterations': 0,
            }

        n_init_boost = min(max_init_boost, max(min_init_boost, train_x_init.shape[0] // ratio_init_boost))
        n_init_boost = round(n_init_boost)
        torch.set_default_dtype(torch.double)

        # calculate y values corresponding to train_x_init
        train_x_init = train_x_init.to(self.device)
        if train_y_init is not None:
            train_y_init = train_y_init.to(self.device)
            full_y = train_y_init.to(dtype=train_x_init.dtype, device=self.device)
        else:
            full_y = objective(train_x_init).to(dtype=train_x_init.dtype, device=self.device)

        # set target value as the 5th percentile of the y values
        sorted_y, _ = torch.sort(full_y)
        percentile = max(1, round(len(sorted_y) * 0.05)) # We use the 5th percentile point as the target value; if it is the global minimum, the second-best point is used instead.
                                                        # Change this to percentile = 0 to use global optimum as target value
        while True:
            target = sorted_y[percentile].item()
            # remove the point less than or equal to target for construction of initial sample for internal BO process
            valid_mask = full_y > target
            valid_x = train_x_init[valid_mask]
            if valid_x.shape[0] >= n_init_boost:
                break
            else:
                percentile -= 1
                if percentile < 0:
                    raise ValueError("Insufficient data")
        # construction of initial sample for internal BO process
        selected_indices = self.select_representative_samples(valid_x, n_init_boost) # To randomly select r_n istead of Kmeans-clustering,
                                                                                    # selected_indices = torch.randperm(valid_x.shape[0], device=self.device)[:n_init_boost]
        valid_indices = torch.arange(train_x_init.shape[0], device=self.device)[valid_mask]
        train_indices= valid_indices[selected_indices]
        selected_train_x_init = train_x_init[train_indices]

        # leftover points for candidate set: treat as undiscovered points
        candidate_mask = torch.ones(train_x_init.shape[0], dtype=torch.bool, device=self.device)
        candidate_mask[train_indices] = False
        self.filtered_candidate_x = train_x_init[candidate_mask]
        if train_y_init is not None:
            selected_train_y_init = train_y_init[train_indices]
            self.filtered_candidate_y = train_y_init[candidate_mask]

        # Parallelize the evaluation of kernel-acquisition combinations
        combinations = [
            (acq, kern)
            for acq in self.acquisition_candidates
            for kern in self.kernel_candidates
        ]
        n_combinations = len(combinations)
        n_workers = min(10, max(8, n_combinations // 2))

        def evaluate_combo(acquisition_type, kernel_type):
            iterations = 0
            train_x = selected_train_x_init.clone()
            filtered_candidate_x = self.filtered_candidate_x.clone()
            if train_y_init is not None:
                train_y = selected_train_y_init.clone()
                filtered_candidate_y = self.filtered_candidate_y.clone()
            else:
                train_y = objective(train_x).to(dtype=train_x.dtype, device=self.device)
                filtered_candidate_y = None

            while train_x.shape[0] < train_x_init.shape[0]:
                iterations += 1
                if iterations > max_iter_boost:
                    break
                fixed_hp_state_for_kernel = (
                    self.fixed_hp_states.get(kernel_type)
                    if (self.fixed_hp_mode and self.fixed_hp_states is not None)
                    else None
                )
                next_x, next_y, next_idx = self.get_next_point(
                    train_x=train_x,
                    train_y=train_y,
                    filtered_candidate_x=filtered_candidate_x,
                    filtered_candidate_y=filtered_candidate_y,
                    kernel_type=kernel_type,
                    acquisition_type=acquisition_type,
                    objective=objective,
                    fixed_hp_state=fixed_hp_state_for_kernel,
                    fixed_hp_normalize_stats=self.fixed_hp_normalize_stats if self.fixed_hp_mode else None,
                )
                train_x = torch.cat([train_x, next_x], dim=0)
                train_y = torch.cat([train_y, next_y], dim=0)

                mask = torch.ones(filtered_candidate_x.shape[0], dtype=torch.bool, device=self.device)
                mask[next_idx] = False
                filtered_candidate_x = filtered_candidate_x[mask]
                if self.is_fixed_candidate_x:
                    filtered_candidate_y = filtered_candidate_y[mask]
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stopping criterion
                if train_y.min().item() <= target:
                    break

            del train_x, train_y, filtered_candidate_x, filtered_candidate_y
            gc.collect()

            return {
                'kernel': kernel_type.value,
                'acquisition': acquisition_type.value,
                'iterations': iterations,
            }
        
        # Parallel execution of the evaluation function for each combination
        with Parallel(n_jobs=n_workers) as parallel: 
            results = parallel(
                delayed(evaluate_combo)(acq, kern)
                for acq, kern in combinations
            )


        # The kernel-acquisition pair that achieves the fastest convergence is selected
        min_result = min(results, key=lambda x: x['iterations']) # For random tie-breaking rule,
                                                                # self.set_seed(seed)
                                                                # min_iter = min(r['iterations'] for r in results)
                                                                # min_results = [r for r in results if r['iterations'] == min_iter]
                                                                # min_result = random.choice(min_results)
        return {
            'recommended_kernel': min_result['kernel'],
            'recommended_acquisition': min_result['acquisition'],
            'iterations': min_result['iterations']
        }

    def get_kernel_acq(self, train_x, train_y, objective, iter, seed, n_init_points, base_dir):
        train_x = train_x.to(self.device)
        if train_y is not None:
            train_y = train_y.to(self.device)

        # Get the recommended kernel and acquisition type
        recommended = self.recommend(
            train_x_init=train_x,
            train_y_init=train_y,
            objective=objective,
            seed=iter,
        )

        kernel_type = KernelType(recommended['recommended_kernel'])
        acquisition_type = AcquisitionType(recommended['recommended_acquisition'])

        # Save the recommendation log
        save_recommendation_log(
            objective_name=objective if isinstance(objective, str) else objective.__name__,
            seed=seed,
            kernel=recommended['recommended_kernel'],
            acquisition=recommended['recommended_acquisition'],
            n_init_sample=n_init_points,
            iteration=iter,
            base_dir=base_dir
        )
        return kernel_type, acquisition_type

    def select_representative_samples(self, x_cand, n_select):
        x_cand_np = x_cand.detach().cpu().numpy() if isinstance(x_cand, torch.Tensor) else x_cand

        # Use KMeans to cluster the candidate points into n_select clusters
        kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(x_cand_np)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, device=self.device)
        cluster_labels_tensor = torch.tensor(cluster_labels, device=self.device)

        # Find the indices of the closest points to the cluster centers
        selected_indices = []
        for i in range(n_select):
            cluster_members = torch.where(cluster_labels_tensor == i)[0]
            distances = torch.norm(x_cand[cluster_members] - cluster_centers[i], dim=1)
            closest_idx = cluster_members[torch.argmin(distances)]
            selected_indices.append(closest_idx.item())
        selected_indices = torch.tensor(selected_indices, device=self.device)

        return selected_indices
