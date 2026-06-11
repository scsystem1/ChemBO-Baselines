from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BenchmarkConfig:
    bounds: tuple
    n_grid: int
    dim: int
    target: float


class Benchmarks:
    # Global minimum: f(0, 0, ..., 0) = 0
    ACKLEY_CONFIG = BenchmarkConfig(
        bounds=(-31.5, 31.5),
        n_grid=37,
        dim=4,
        target=0.0
    )

    # Global minimum: f(1,1,...,1) = 0
    LEVY_CONFIG = BenchmarkConfig(
        bounds=(-10, 10),
        n_grid=41,
        dim=4,
        target=0.0
    )

    # Global minimum: f(1,1,...,1) = 0
    ROSENBROCK_CONFIG = BenchmarkConfig(
        bounds=(-5, 10),
        n_grid=31,
        dim=4,
        target=0.0
    )

    @staticmethod
    def Ackley(individuals):
        """
        Ackley function
        Global minimum: f(0,0,...,0) = 0
        """
        if isinstance(individuals, torch.Tensor):
            individuals = individuals.detach().cpu().numpy()
        else:
            individuals = np.asarray(individuals)
        n = individuals.shape[1]
        sum1 = np.sum(individuals ** 2, axis=1)
        sum2 = np.sum(np.cos(2 * np.pi * individuals), axis=1)

        ackley_value = -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.exp(1)
        return torch.tensor(ackley_value)

    @staticmethod
    def Levy(individuals):
        """
        Levy function
        Global minimum: f(1,1,...,1) = 0
        """
        if isinstance(individuals, torch.Tensor):
            individuals = individuals.detach().cpu().numpy()
        else:
            individuals = np.asarray(individuals)
        w = 1 + (individuals - 1) / 4

        term1 = np.sin(np.pi * w[:, 0]) ** 2
        term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)

        sum_term = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)

        return torch.tensor(term1 + sum_term + term3)

    @staticmethod
    def Rosenbrock(individuals):
        """
        Rosenbrock function (Banana function)
        Global minimum: f(1,1,...,1) = 0
        """
        if isinstance(individuals, torch.Tensor):
            individuals = individuals.detach().cpu().numpy()
        else:
            individuals = np.asarray(individuals)
        sum_term = np.sum(100.0 * (individuals[:, 1:] - individuals[:, :-1] ** 2) ** 2 +
                          (individuals[:, :-1] - 1) ** 2, axis=1)
        return torch.tensor(sum_term)

