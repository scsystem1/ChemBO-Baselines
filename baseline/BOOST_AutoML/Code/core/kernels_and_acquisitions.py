import warnings
from enum import Enum

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, RQKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.utils.warnings import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class KernelType(Enum):
    RBF = "RBF"
    MATERN32 = "Matern32"
    MATERN52 = "Matern52"
    RQ = "RQ"
    TBD = "TBD"

class AcquisitionType(Enum):
    EI = "EI"
    PI = "PI"
    UCB = "UCB"
    PM = "PM"
    TBD = "TBD"

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type, lengthscale_constraint, outputscale_constraint):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        # Base kernel selection
        if kernel_type == KernelType.RBF:
            base_kernel = RBFKernel(lengthscale_constraint=lengthscale_constraint)
        elif kernel_type == KernelType.MATERN32:
            base_kernel = MaternKernel(nu=1.5, lengthscale_constraint=lengthscale_constraint)
        elif kernel_type == KernelType.MATERN52:
            base_kernel = MaternKernel(nu=2.5, lengthscale_constraint=lengthscale_constraint)
        elif kernel_type == KernelType.RQ:
            base_kernel = RQKernel(lengthscale_constraint=lengthscale_constraint)
            base_kernel.raw_alpha = torch.nn.Parameter(torch.tensor(2.0, device=train_x.device, dtype=train_x.dtype))
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)
        self.to(train_x.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


