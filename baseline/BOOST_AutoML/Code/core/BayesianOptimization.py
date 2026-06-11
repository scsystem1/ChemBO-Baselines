import gc
import math
import random

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood

from core.kernels_and_acquisitions import AcquisitionType, GPModel


class BayesianOptimizer:
    def __init__(self, device='cpu'):
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

    @staticmethod
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

    def get_next_point(self, train_x, train_y, filtered_candidate_x, filtered_candidate_y, kernel_type, acquisition_type, objective=None,
                       fixed_hp_state=None, fixed_hp_normalize_stats=None):
        # Default normalize: x is min-max normalized to [0, 1], y is standardized with median and std.
        # When fixed_hp_normalize_stats is provided, reuse the
        # external full-data normalization so the cached hyperparameters keep their semantics.
        if fixed_hp_normalize_stats is not None:
            x_min = fixed_hp_normalize_stats['x_min']
            x_range = fixed_hp_normalize_stats['x_range']
            y_median = fixed_hp_normalize_stats['y_median']
            y_std = fixed_hp_normalize_stats['y_std']
            train_x_normalized = (train_x - x_min) / x_range
            train_y_normalized = (train_y - y_median) / y_std
        else:
            x_min, x_range, y_median, y_std, train_x_normalized, train_y_normalized = self.normalize_data(train_x, train_y)
        candidate_x_normalized = (filtered_candidate_x - x_min) / x_range

        # Generate and train GP model. fixed_hp_state (kernel-specific state_dict from a fit
        # on the full external dataset) skips the Adam loop and freezes hyperparameters.
        if fixed_hp_state is not None:
            model, likelihood = self._build_fixed_hp_model(
                train_x_normalized=train_x_normalized,
                train_y_normalized=train_y_normalized,
                kernel_type=kernel_type,
                fixed_hp_state=fixed_hp_state,
            )
        else:
            model, likelihood = self._train_model(train_x_normalized=train_x_normalized, train_y_normalized=train_y_normalized, kernel_type=kernel_type)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get predictions for the candidate points and find the next point
            observed_pred = likelihood(model(candidate_x_normalized))
            best_f = train_y.min().item()
            next_x_idx = self._get_next_idx(acquisition_type=acquisition_type, best_f=best_f, observed_pred=observed_pred, y_median=y_median, y_std=y_std)
            next_x = filtered_candidate_x[next_x_idx].unsqueeze(0)

            # Generate train_y
            if filtered_candidate_y is not None:
                next_y = filtered_candidate_y[next_x_idx].unsqueeze(0).to(self.device)
            else:
                next_y = objective(next_x).to(dtype=next_x.dtype)

        # Remove unnecessary variables to free memory
        del candidate_x_normalized, observed_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return next_x, next_y, next_x_idx

    @staticmethod
    def normalize_data(train_x, train_y):
        """
        Normaize data: x is min-max normalized to [0, 1], y is standardized with median and std
        """
        # x is min-max normalized to [0, 1]
        x_min = train_x.min(dim=0)[0]
        x_max = train_x.max(dim=0)[0]
        x_range = torch.clamp(x_max - x_min, min=1e-8)
        train_x_normalized = (train_x - x_min) / x_range

        # y is standardized with median and std
        y_median = train_y.median()
        y_std = train_y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1e-6)
        train_y_normalized = (train_y - y_median) / y_std

        return x_min, x_range, y_median, y_std, train_x_normalized, train_y_normalized

    @staticmethod
    def _make_constraints(input_dim):
        noise_constraint = Interval(5e-4, 0.2)
        lengthscale_constraint = Interval(5 * 1e-6, math.sqrt(input_dim))
        outputscale_constraint = Interval(0.05, 20.0)
        return noise_constraint, lengthscale_constraint, outputscale_constraint

    @staticmethod
    def _train_model(train_x_normalized, train_y_normalized, kernel_type):
        noise_constraint, lengthscale_constraint, outputscale_constraint = (
            BayesianOptimizer._make_constraints(train_x_normalized.shape[1])
        )

        # GP Model
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x_normalized.device, dtype=train_y_normalized.dtype)
        model = GPModel(train_x_normalized, train_y_normalized, likelihood, kernel_type=kernel_type,  lengthscale_constraint=lengthscale_constraint, outputscale_constraint=outputscale_constraint)

        # Model training
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        lr = 0.05
        max_iter = 50
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            output = model(train_x_normalized)
            loss = -mll(output, train_y_normalized)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()

        return model, likelihood

    @staticmethod
    def fit_full_data_gp(train_x, train_y, kernel_type):
        """Fit a GP on the full external dataset and return (model, likelihood, normalize_stats).

        Used by the fixed-hyperparameter ablation path to extract reference hyperparameters
        (lengthscale, outputscale, noise) per kernel before delegating to BOOST. The same
        normalize_stats must be reused inside the internal simulation so the cached
        hyperparameters keep their geometric meaning.
        """
        x_min, x_range, y_median, y_std, train_x_normalized, train_y_normalized = (
            BayesianOptimizer.normalize_data(train_x, train_y)
        )
        model, likelihood = BayesianOptimizer._train_model(
            train_x_normalized=train_x_normalized,
            train_y_normalized=train_y_normalized,
            kernel_type=kernel_type,
        )
        normalize_stats = {
            'x_min': x_min.detach().clone(),
            'x_range': x_range.detach().clone(),
            'y_median': y_median.detach().clone() if torch.is_tensor(y_median) else torch.tensor(y_median),
            'y_std': y_std.detach().clone() if torch.is_tensor(y_std) else torch.tensor(y_std),
        }
        return model, likelihood, normalize_stats

    @staticmethod
    def _build_fixed_hp_model(train_x_normalized, train_y_normalized, kernel_type, fixed_hp_state):
        """Build a GP whose kernel/noise hyperparameters are pinned to fixed_hp_state.

        fixed_hp_state holds the GPModel and GaussianLikelihood state_dicts produced by
        fit_full_data_gp on the external full data. We instantiate a fresh model on the
        reference set, load the state, and disable gradients so subsequent evaluation
        only re-computes K(X*, X*) at the cached hyperparameters -- no MLL optimization.
        """
        noise_constraint, lengthscale_constraint, outputscale_constraint = (
            BayesianOptimizer._make_constraints(train_x_normalized.shape[1])
        )
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
            device=train_x_normalized.device, dtype=train_y_normalized.dtype
        )
        model = GPModel(
            train_x_normalized,
            train_y_normalized,
            likelihood,
            kernel_type=kernel_type,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
        )

        # Transfer hyperparameters (lengthscale, outputscale, kernel-specific raw params,
        # constant mean, noise) from the full-data fit. strict=False because the
        # train_inputs/train_targets buffers differ in shape between full and reference set.
        model.load_state_dict(fixed_hp_state['model'], strict=False)
        likelihood.load_state_dict(fixed_hp_state['likelihood'], strict=False)

        # Re-bind the reference set as the conditioning data: load_state_dict above only
        # patched parameter buffers, but ExactGP keeps train_inputs/train_targets that
        # must match the reference set so K(X, X) is rebuilt on the subset.
        model.set_train_data(inputs=train_x_normalized, targets=train_y_normalized, strict=False)

        # Freeze every parameter so any downstream code path cannot accidentally optimize.
        for param in model.parameters():
            param.requires_grad_(False)
        for param in likelihood.parameters():
            param.requires_grad_(False)

        return model, likelihood

    def _get_next_idx(self, acquisition_type, best_f, observed_pred, y_median, y_std):
        # Denormalize predictions
        # Assume minimization problem. Should be modified if applied to maximization problem
        mean = observed_pred.mean * y_std + y_median
        stddev = observed_pred.stddev * y_std
        if acquisition_type == AcquisitionType.EI:
            acq_values = self._expected_improvement(best_f=best_f, mean=mean, sigma=stddev)
            next_x_idx = torch.argmax(acq_values)
        elif acquisition_type == AcquisitionType.PI:
            acq_values = self._probability_improvement(best_f=best_f, mean=mean, sigma=stddev)
            next_x_idx = torch.argmax(acq_values)
        elif acquisition_type == AcquisitionType.PM:
            acq_values = self._posterior_mean(mean=mean)
            next_x_idx = torch.argmin(acq_values)
        elif acquisition_type == AcquisitionType.UCB:
            acq_values = self._upper_confidence_bound(mean=mean, sigma=stddev)
            next_x_idx = torch.argmin(acq_values)
        else:
            raise ValueError("Unsupported acquisition type")

        return next_x_idx


    @staticmethod
    def _expected_improvement(best_f, mean, sigma, epsilon=0):
        """Expected Improvement acquisition function"""
        with torch.no_grad():
            z = (best_f - mean - epsilon) / sigma
            cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
            pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
            return (best_f - mean - epsilon) * cdf + sigma * pdf

    @staticmethod
    def _probability_improvement(best_f, mean, sigma, epsilon=0):
        """Probability of Improvement acquisition function"""
        with torch.no_grad():
            z = (best_f - mean - epsilon) / sigma
            return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    @staticmethod
    def _upper_confidence_bound(mean, sigma, kappa=0.1):
        """Upper Confidence Bound acquisition function"""
        with torch.no_grad():
            return mean - kappa * sigma

    @staticmethod
    def _posterior_mean(mean):
        """PM (Posterior Mean) acquisition function"""
        return mean
