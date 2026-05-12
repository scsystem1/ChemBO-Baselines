import numpy as np
import math
import torch
from botorch.models import SingleTaskGP
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
import gpytorch
from botorch.acquisition.analytic import (
    ProbabilityOfImprovement, 
    LogProbabilityOfImprovement,
    ExpectedImprovement, 
    LogExpectedImprovement, 
    UpperConfidenceBound,
    PosteriorMean,
    PosteriorStandardDeviation
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.utils import get_optimal_samples
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.optim import (
    optimize_acqf,
)
from torch.quasirandom import SobolEngine
from botorch.acquisition.objective import (
    ScalarizedPosteriorTransform,
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _fit_gp_once(X, Y, *, fit_device, jitter):
    nrows = X.shape[0]
    ncols = X.shape[1]
    X_fit = X.detach().clone().to(device=fit_device, dtype=dtype).reshape((nrows, ncols))
    Y_fit = Y.detach().clone().to(device=fit_device, dtype=dtype).reshape((nrows, 1))
    gp = SingleTaskGP(
        X_fit,
        Y_fit,
        covar_module=ScaleKernel(
            MaternKernel(
                ard_num_dims=X_fit.shape[-1]
            )
        ),
        input_transform=Normalize(d=X_fit.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    with gpytorch.settings.cholesky_jitter(float(jitter)):
        fit_gpytorch_mll(mll)
    return gp


def fit_gp(
    X,
    Y,
    *,
    return_diagnostics=False,
    jitter_values=(1e-6, 1e-4, 1e-2),
    retries_per_config=2,
):
    output_device = X.device
    fit_devices = [output_device]
    if output_device.type != "cpu":
        fit_devices.append(torch.device("cpu"))

    attempts = []
    last_error = None

    for fit_device in fit_devices:
        for jitter in jitter_values:
            for retry_idx in range(retries_per_config):
                try:
                    gp = _fit_gp_once(X, Y, fit_device=fit_device, jitter=jitter)
                    if fit_device != output_device:
                        gp = gp.to(output_device)
                    diagnostics = {
                        "status": "success",
                        "fit_device": str(fit_device),
                        "return_device": str(output_device),
                        "jitter": float(jitter),
                        "retry_index": int(retry_idx),
                        "attempt_count": int(len(attempts) + 1),
                        "used_cpu_fallback": fit_device.type == "cpu",
                        "used_jitter_fallback": float(jitter) != float(jitter_values[0]),
                        "attempts": attempts
                        + [
                            {
                                "fit_device": str(fit_device),
                                "jitter": float(jitter),
                                "retry_index": int(retry_idx),
                                "status": "success",
                                "error": None,
                            }
                        ],
                    }
                    if return_diagnostics:
                        return gp, diagnostics
                    return gp
                except Exception as exc:
                    last_error = exc
                    attempts.append(
                        {
                            "fit_device": str(fit_device),
                            "jitter": float(jitter),
                            "retry_index": int(retry_idx),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    diagnostics = {
        "status": "failed",
        "fit_device": None,
        "return_device": str(output_device),
        "jitter": None,
        "retry_index": None,
        "attempt_count": int(len(attempts)),
        "used_cpu_fallback": any(attempt["fit_device"] == "cpu" for attempt in attempts),
        "used_jitter_fallback": any(attempt["jitter"] != float(jitter_values[0]) for attempt in attempts),
        "attempts": attempts,
    }
    message = "All attempts to fit the model have failed."
    if last_error is not None:
        message = f"{message} Last error: {last_error}"
    error = ModelFittingError(message)
    error.fit_diagnostics = diagnostics
    raise error from last_error

def get_negated_transform():
    return ScalarizedPosteriorTransform(
        weights=torch.tensor([-1.0], device=device, dtype=dtype)
    )

def create_minimized_acquisition_function(acq_type_class, model, **kwargs):
    transform = get_negated_transform()
    class MinimizedAcquisitionFunction(acq_type_class):
        def __init__(self):
            super().__init__(model=model, posterior_transform=transform, **kwargs)

    acqf = MinimizedAcquisitionFunction()

    return acqf

def _prepare_acquisition_function(acq_type, bounds, best_f, gp):
    if acq_type == "PI":
        acq_func = ProbabilityOfImprovement(model=gp, best_f=best_f, maximize=False)
    elif acq_type == "LogPI":
        acq_func = LogProbabilityOfImprovement(model=gp, best_f=best_f, maximize=False)
    elif acq_type == "EI":
        acq_func = ExpectedImprovement(model=gp, best_f=best_f, maximize=False)
    elif acq_type == "LogEI":
        acq_func = LogExpectedImprovement(model=gp, best_f=best_f, maximize=False)
    elif acq_type == "UCB":
        N = gp.train_inputs[0].shape[0]
        acq_func = UpperConfidenceBound(model=gp, beta=4*math.log(N), maximize=False)
    elif acq_type == "PosMean":
        acq_func = PosteriorMean(model=gp, maximize=False)
    elif acq_type == "PosSTD":
        acq_func = PosteriorStandardDeviation(model=gp, maximize=False)
    elif acq_type == "qKG":
        acq_func = create_minimized_acquisition_function(qKnowledgeGradient, model=gp, num_fantasies=4)
    elif acq_type == "qPES":
        optimal_inputs, _ = get_optimal_samples(
            model=gp.cpu(), 
            bounds=bounds.cpu(), 
            num_optima=4,
            posterior_transform=get_negated_transform().cpu(),
        )        
        acq_func = qPredictiveEntropySearch(
            model=gp.to(device), 
            maximize=False,
            optimal_inputs=optimal_inputs.to(device), 
        )
    elif acq_type == "qJES":
        optimal_inputs, optimal_outputs = get_optimal_samples(
            model=gp.cpu(), 
            bounds=bounds.cpu(), 
            num_optima=4,
            posterior_transform=get_negated_transform().cpu(),
        )
        acq_func = qJointEntropySearch(
            model=gp.to(device),
            optimal_inputs=optimal_inputs.to(device).to(dtype),
            optimal_outputs=optimal_outputs.to(device).to(dtype),
            posterior_transform=get_negated_transform(),
            estimation_type="LB",
        )
    elif acq_type == "qMES":
        acq_func = qLowerBoundMaxValueEntropy(
            model=gp,
            candidate_set=torch.rand(1000, bounds.size(1)).to(dtype).to(device),
            maximize=False,
        )
    elif acq_type == "TS":
        acq_func = create_minimized_acquisition_function(MaxPosteriorSampling, model=gp)
    else:
        print(f"Invalid acquisition function type: {acq_type}")
        raise ValueError("Invalid acquisition function type")    
    return acq_func

def _optimize_acqf(acq_type, acq_func, bounds, cheap=False):
    if acq_type == "TS":
        n_candidates = min(5000, max(2000, 200 * bounds.size(1))) if not cheap else 50
        sobol = SobolEngine(bounds.size(1), scramble=True)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_candidates).to(dtype=dtype, device=device)
        candidate = acq_func(X_cand, num_samples=1)
    elif acq_type == "qPES":
        n_candidates = min(1000, max(1000, 200 * bounds.size(1))) if not cheap else 50
        sobol = SobolEngine(bounds.size(1), scramble=True)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_candidates).to(dtype=dtype, device=device)
        acq_val = acq_func(X_cand)
        candidate = X_cand[torch.argmax(acq_val)].unsqueeze(0)
    else:
        optimize_acqf_kwargs = {
            "acq_function": acq_func,
            "bounds": bounds,
            "q": 1,
            "num_restarts": 10 if not cheap else 1,
            "raw_samples": 512 if not cheap else 16
        }
        candidate, _ = optimize_acqf(**optimize_acqf_kwargs)
    return candidate

def bo_single_iteration(
    train_X,
    train_Y,
    acq_type,
    objective_func,
    bounds
):
    """
    Performs a single Bayesian Optimization iteration for a given acquisition function type.
    This function is primarily used by individual (non-portfolio) BO strategies.
    For GP-Hedge, a modified version `bo_single_iteration_gph` is used to get nominated points.
    """
    gp = fit_gp(train_X, train_Y)
    acq_func = _prepare_acquisition_function(acq_type, bounds, train_Y.min(), gp)
    # Optimize the acquisition function to find the next query point
    candidate = _optimize_acqf(acq_type, acq_func, bounds)
    # Evaluate the function at the new point
    new_Y = objective_func(candidate).unsqueeze(-1)
    # Update the dataset
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_Y])
    return train_X, train_Y, gp

def calculate_cumulative_regret(observations, true_minimum):
    """
    Calculate the cumulative regret over the iterations.
    
    Args:
        observations: numpy array of observed values at each iteration
        true_minimum: float, true global minimum of the objective function
        
    Returns:
        Cumulative regret as a numpy array
    """
    # Calculate cumulative regret
    cumulative_regret = np.cumsum(observations - true_minimum)
    
    return cumulative_regret
