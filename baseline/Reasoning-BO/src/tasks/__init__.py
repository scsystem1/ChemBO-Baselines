from .chemistry.chemistry import ChemistryMetric
from .chemistry.tabular import TabularChemistryMetric
from .maths.hartmann6 import Hartmann6Metric
from .maths.ackley import AckleyMetric
from .maths.levy import LevyMetric
from .maths.rosenbrock import RosenbrockMetric

try:
    from .lunar.lunar import LunarLanderMetric
except Exception:  # Optional dependency: Box2D / gym
    LunarLanderMetric = None


__all__ = [
    'ChemistryMetric',
    'TabularChemistryMetric',
    'Hartmann6Metric',
    'AckleyMetric',
    'LevyMetric',
    'RosenbrockMetric',
]

if LunarLanderMetric is not None:
    __all__.append('LunarLanderMetric')
