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

    def get_next_point(self, train_x, train_y, filtered_candidate_x, filtered_candidate_y, kernel_type, acquisition_type, objective=None):
        # Normaize data: x is min-max normalized to [0, 1], y is standardized with median and std
        x_min, x_range, y_median, y_std, train_x_normalized, train_y_normalized = self.normalize_data(train_x, train_y)
        candidate_x_normalized = (filtered_candidate_x - x_min) / x_range

        # Generate and train GP model
        model, likelihood = self._train_model(train_x_normalized=train_x_normalized, train_y_normalized=train_y_normalized, kernel_type=kernel_type)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get predictions for the candidate points and find the next point
            observed_pred = likelihood(model(candidate_x_normalized))
            best_f = train_y.min().item()
            next_x_idx, max_acq_value = self._get_next_idx(acquisition_type=acquisition_type, best_f=best_f, observed_pred=observed_pred, y_median=y_median, y_std=y_std)
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

        return next_x, next_y, next_x_idx, max_acq_value

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
    def _train_model(train_x_normalized, train_y_normalized, kernel_type):
        # Constraints for the GP model
        noise_constraint = Interval(5e-4, 0.2)
        lengthscale_constraint = Interval(5*1e-6, math.sqrt(train_x_normalized.shape[1]))
        outputscale_constraint = Interval(0.05, 20.0)

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

    def _get_next_idx(self, acquisition_type, best_f, observed_pred, y_median, y_std):
        # Denormalize predictions
        # Assume minimization problem. Should be modified if applied to maximization problem
        mean = observed_pred.mean * y_std + y_median
        stddev = observed_pred.stddev * y_std
        if acquisition_type == AcquisitionType.EI:
            acq_values = self._expected_improvement(best_f=best_f, mean=mean, sigma=stddev)
            next_x_idx = torch.argmax(acq_values)
            max_acq_value = acq_values[next_x_idx]
        elif acquisition_type == AcquisitionType.PI:
            acq_values = self._probability_improvement(best_f=best_f, mean=mean, sigma=stddev)
            next_x_idx = torch.argmax(acq_values)
            max_acq_value = acq_values[next_x_idx]
        elif acquisition_type == AcquisitionType.PM:
            acq_values = self._posterior_mean(mean=mean)
            next_x_idx = torch.argmin(acq_values)
            max_acq_value = -acq_values[next_x_idx]
        elif acquisition_type == AcquisitionType.UCB:
            acq_values = self._upper_confidence_bound(mean=mean, sigma=stddev)
            next_x_idx = torch.argmin(acq_values)
            max_acq_value = -acq_values[next_x_idx]
        else:
            raise ValueError("Unsupported acquisition type")

        return next_x_idx, max_acq_value


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