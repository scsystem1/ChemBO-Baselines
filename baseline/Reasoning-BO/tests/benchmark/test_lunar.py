import pytest
import numpy as np
from ax import (
    Experiment,
    Objective,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    Runner,
    SearchSpace,
)
from src.tasks import LunarLanderMetric
from src.tasks.lunar.lunar_utils import simulate_lunar_lander
from ax.core import Arm


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


@pytest.fixture
def lunar_search_space():
    """创建Lunar Lander的搜索空间 (12个参数)"""
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(12)
        ]
    )
    # for faithful bo: meaningful parameter names
    # return SearchSpace(
    #     parameters=[
    #         # 推进系统参数
    #         RangeParameter(
    #             name="main_engine_throttle",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="主发动机推力控制系数",
    #         ),
    #         RangeParameter(
    #             name="side_engine_throttle",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="侧向发动机推力控制系数",
    #         ),
    #         # 控制增益参数
    #         RangeParameter(
    #             name="hover_control_gain",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="悬停控制PID增益",
    #         ),
    #         RangeParameter(
    #             name="attitude_control_gain",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="姿态稳定控制增益",
    #         ),
    #         # 着陆机构参数
    #         RangeParameter(
    #             name="landing_leg_damping",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="着陆腿阻尼系数",
    #         ),
    #         RangeParameter(
    #             name="safety_margin",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="着陆安全余量阈值",
    #         ),
    #         # 飞行限制参数
    #         RangeParameter(
    #             name="angular_rate_limit",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="最大允许角速度(rad/s)",
    #         ),
    #         RangeParameter(
    #             name="vertical_speed_limit",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="最大垂直下降速度(m/s)",
    #         ),
    #         RangeParameter(
    #             name="horizontal_speed_limit",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="最大水平速度(m/s)",
    #         ),
    #         # 辅助系统参数
    #         RangeParameter(
    #             name="velocity_smoothing",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="速度测量平滑因子",
    #         ),
    #         RangeParameter(
    #             name="fuel_consumption_factor",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="燃料消耗率调整系数",
    #         ),
    #         RangeParameter(
    #             name="emergency_override",
    #             parameter_type=ParameterType.FLOAT,
    #             lower=0.0,
    #             upper=1.0,
    #             description="紧急控制切换阈值",
    #         ),
    #     ]
    # )


@pytest.fixture
def lunar_optimization_config():
    """创建优化配置"""
    return OptimizationConfig(
        objective=Objective(metric=LunarLanderMetric(name="lunar"))
    )


def test_single_simulation():
    """测试单个模拟运行"""
    x = np.random.rand(12)
    reward = simulate_lunar_lander([x, 42])  # 固定种子
    assert isinstance(reward, float)
    assert -300 <= reward <= 300  # Lunar Lander的典型奖励范围


def test_metric_evaluation(lunar_search_space):
    """测试Metric类的评估功能"""
    metric = LunarLanderMetric(
        name="test", seed=range(3)
    )  # 使用3个种子减少计算量

    # 测试参数评估
    params = {f"x{i+1}": 0.5 for i in range(12)}  # 中性参数
    reward = metric._evaluate_lander(params)
    assert isinstance(reward, float)

    # 测试完整试验数据获取
    exp = Experiment(
        name="test_lunar",
        search_space=lunar_search_space,
    )
    trial = exp.new_trial().add_arm(Arm(parameters=params))
    data = metric.fetch_trial_data(trial)
    print(data)
    assert data.is_ok()


@pytest.mark.slow
def test_full_optimization(lunar_search_space, lunar_optimization_config):
    """完整BO流程测试（标记为slow因为需要较长时间）"""
    exp = Experiment(
        name="lunar_bo_test",
        search_space=lunar_search_space,
        runner=MyRunner(),
        optimization_config=lunar_optimization_config,
    )
    print(exp.fetch_trials_data)

    from src.bo.models import BOModel

    bo_model = BOModel(exp)
    bo_model.hot_start(exp)
    for i in range(3):  # 3个BO迭代（测试用少量迭代）
        trial = exp.new_trial(generator_run=bo_model.gen(1))
        trial.run()

    # 验证结果
    assert len(exp.trials) == 8  # 5 sobol + 3 bo
    best_arm = exp.fetch_data().df.loc[exp.fetch_data().df["mean"].idxmax()]
    assert best_arm["mean"] > -250  # 应优于随机策略
