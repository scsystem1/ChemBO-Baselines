import pytest
import numpy as np
from ax import (
    Experiment,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.core import Arm
from src.tasks import Hartmann6Metric


@pytest.fixture
def hartmann6_search_space():
    """Hartmann6固定6维搜索空间"""
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(6)
        ]
    )


def test_hartmann6_optimal():
    """测试已知最优点的评估"""
    metric = Hartmann6Metric(name="test")
    # Hartmann6的已知最优点坐标
    optimal_params = {
        'x1': 0.20169,
        'x2': 0.15001,
        'x3': 0.476874,
        'x4': 0.275332,
        'x5': 0.311652,
        'x6': 0.6573,
    }
    value = metric._evaluate_hartmann6(optimal_params)
    assert pytest.approx(value, abs=1e-3) == -3.32237


def test_hartmann6_boundaries(hartmann6_search_space):
    """测试边界值评估"""
    metric = Hartmann6Metric(name="test_boundary")
    exp = Experiment(
        name="hartmann6_boundary_test",
        search_space=hartmann6_search_space,
    )
    # 测试全0和全1参数
    for val in [0.0, 1.0]:
        trial = exp.new_trial().add_arm(
            Arm(parameters={f"x{i+1}": val for i in range(6)})
        )
        data = metric.fetch_trial_data(trial)
        print(data)
        assert data.is_ok()
