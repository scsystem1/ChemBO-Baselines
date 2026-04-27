import pytest
from ax import (
    Experiment,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.core import Arm
from src.tasks.maths.levy import LevyMetric


@pytest.fixture
def levy_search_space(dim=2):
    """Levy函数搜索空间"""
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=-10.0,
                upper=10.0,
            )
            for i in range(dim)
        ]
    )


def test_levy_optimal():
    """测试Levy函数在全局最小点的值"""
    metric = LevyMetric(name="test_optimal")
    optimal_params = {
        f"x{i+1}": 1.0 for i in range(2)
    }  # 全局最小点在(1,...,1)
    value = metric._evaluate_levy(optimal_params)
    assert pytest.approx(value, abs=1e-6) == 0.0


def test_levy_symmetry(levy_search_space):
    """测试Levy函数的对称性"""
    metric = LevyMetric(name="test_symmetry")
    exp = Experiment(
        name="levy_symmetry_test",
        search_space=levy_search_space,
    )

    # 测试对称点应具有相同值
    params1 = {'x1': 5.0, 'x2': 5.0}  # w = [2.0, 2.0]
    params2 = {'x1': -3.0, 'x2': -3.0}  # w = [0.0, 0.0]（关于w=1对称）

    trial1 = exp.new_trial().add_arm(Arm(parameters=params1))
    trial2 = exp.new_trial().add_arm(Arm(parameters=params2))

    val1 = metric.fetch_trial_data(trial1).ok.df["mean"].iloc[0]
    val2 = metric.fetch_trial_data(trial2).ok.df["mean"].iloc[0]
    assert pytest.approx(val1, abs=1e-6) == val2
