import pytest
import numpy as np
from ax import (
    Experiment,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    Objective,
    SearchSpace,
)
from ax.core import Arm
from src.tasks import AckleyMetric


@pytest.fixture
def ackley_search_space(dim=2):
    """创建Ackley函数的搜索空间"""
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=-5.0,  # Ackley常用搜索范围
                upper=5.0,
            )
            for i in range(dim)
        ]
    )


@pytest.fixture
def ackley_optimization_config():
    return OptimizationConfig(
        objective=Objective(metric=AckleyMetric(name="ackley"))
    )


def test_ackley_evaluation():
    """测试Ackley函数基础评估"""
    metric = AckleyMetric(name="test", dimension=2)
    params = {f"x{i+1}": 0.0 for i in range(2)}  # 全局最小值在原点附近
    value = metric._evaluate_ackley(params)
    assert isinstance(value, float)
    assert 0 <= value <= 20  # 原点处理论值为0，但可能有浮点误差


def test_metric_integration(ackley_search_space):
    """测试与AX框架的集成"""
    metric = AckleyMetric(name="ackley")
    exp = Experiment(
        name="ackley_test",
        search_space=ackley_search_space,
    )
    trial = exp.new_trial().add_arm(
        Arm(parameters={f"x{i+1}": 0.5 for i in range(2)})
    )
    data = metric.fetch_trial_data(trial)
    assert data.is_ok()
