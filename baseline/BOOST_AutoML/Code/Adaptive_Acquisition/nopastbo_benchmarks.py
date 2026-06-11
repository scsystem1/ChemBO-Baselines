import numpy as np

def ackley(x):
    """
    Ackley function (4D version).
    Global minimum: f(x) = 0 at x = [0, 0, 0, 0]
    """
    x = np.atleast_2d(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = x.shape[1]

    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=1) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x), axis=1) / d)

    return (sum_sq_term + cos_term + a + np.exp(1)).reshape(-1, 1)


def levy(x):
    """
    Levy function (4D version).
    Global minimum: f(x) = 0 at x = [1, 1, 1, 1]
    """
    x = np.atleast_2d(x)
    w = 1 + (x - 1) / 4

    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    sum_term = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)

    return (term1 + sum_term + term3).reshape(-1, 1)


def rosenbrock(x):
    """
    Rosenbrock (Banana) function (4D version).
    Global minimum: f(x) = 0 at x = [1, 1, 1, 1]
    """
    x = np.atleast_2d(x)
    sum_term = np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 +
                      (x[:, :-1] - 1) ** 2, axis=1)

    return sum_term.reshape(-1, 1)
