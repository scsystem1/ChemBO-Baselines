import pytest
from ax import (
    Experiment,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.core import Arm
from src.tasks.maths.rosenbrock import RosenbrockMetric


@pytest.fixture
def rosenbrock_search_space(dim=2):
    """Rosenbrock函数搜索空间"""
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=-2.048,
                upper=2.048,  # 经典Rosenbrock搜索范围
            )
            for i in range(dim)
        ]
    )


def test_rosenbrock_optimal():
    """测试全局最小点(1,...,1)"""
    metric = RosenbrockMetric(name="test_optimal")
    optimal_params = {f"x{i+1}": 1.0 for i in range(2)}
    value = metric._evaluate_rosenbrock(optimal_params)
    assert pytest.approx(value, abs=1e-6) == 0.0


def test_rosenbrock_valley(rosenbrock_search_space):
    """测试峡谷特性（2D情况）"""
    metric = RosenbrockMetric(name="test_valley", dimension=2)  # 明确指定2D
    exp = Experiment(
        name="rosenbrock_valley_test",
        search_space=rosenbrock_search_space,
    )

    # 测试点及预期值
    test_cases = [
        (1.0, 1.0, 0.0),  # 全局最小点
        (1.0, 1.5, 25.0),  # 峡谷上方
        (1.0, 0.5, 25.0),  # 峡谷下方
        (1.0, -1.5, 625.0),  # 远离峡谷（应大于100）
    ]

    for x1, x2, expected in test_cases:
        trial = exp.new_trial().add_arm(Arm(parameters={'x1': x1, 'x2': x2}))
        value = metric.fetch_trial_data(trial).ok.df["mean"].iloc[0]
        assert pytest.approx(value, abs=1e-6) == expected


def test_rosenbrock_valley_3d():
    """测试3D情况下的峡谷特性"""
    metric = RosenbrockMetric(name="test_valley_3d", dimension=3)

    # 3D情况下峡谷沿x1=1,x2=1延伸
    params1 = {'x1': 1.0, 'x2': 1.0, 'x3': 1.0}  # 最优线
    params2 = {'x1': 1.0, 'x2': 1.0, 'x3': 1.5}  # 偏离最优线

    assert metric._evaluate_rosenbrock(params1) == 0.0
    assert metric._evaluate_rosenbrock(params2) == 25.0
