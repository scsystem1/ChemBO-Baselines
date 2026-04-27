import pytest
import numpy as np
import os
from ax import (
    ComparisonOp,
    Experiment,
    Objective,
    OptimizationConfig,
    OutcomeConstraint,
    ParameterType,
    RangeParameter,
    SearchSpace,
    Runner,
)
from src.bo.models import BOModel
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.hartmann6 import Hartmann6Metric

from src.utils.metric import save_trial_data, extract_metric

# ---------------------------------- define a runner ----------------------------------


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


@pytest.fixture(scope="module")
def initialize_experiment():
    # 初始化 experiment
    hartmann_search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,
            )
            for i in range(6)
        ]
    )
    param_names = [f"x{i}" for i in range(6)]

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(name="hartman6", param_names=param_names),
            minimize=True,
        ),
        outcome_constraints=[
            OutcomeConstraint(
                metric=L2NormMetric(
                    name="l2norm", param_names=param_names, noise_sd=0.2
                ),
                op=ComparisonOp.LEQ,
                bound=1.25,
                relative=False,
            )
        ],
    )

    exp = Experiment(
        name="test_hartmann",
        search_space=hartmann_search_space,
        optimization_config=optimization_config,
        runner=MyRunner(),
    )

    # 运行 experiment 的 trial
    bo_model = BOModel(exp)
    bo_model.hot_start(exp)
    generator_run = bo_model.gen(n=3)
    trial = exp.new_batch_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

    return exp, trial


# test_metric_utils.py
def test_save_trial_data(initialize_experiment):
    # 定义保存目录和文件名
    experiment, trial = initialize_experiment
    save_dir = "./test_results"
    filename = "test"
    save_trial_data(experiment, trial, save_dir=save_dir, filename=filename)
    # 检查文件是否存在
    saved_file_path = os.path.join(save_dir, f"{filename}.json")
    assert os.path.exists(
        saved_file_path
    ), "The trial data file was not saved."


def test_extract_metric_hartmann_values(initialize_experiment):
    experiment, _ = initialize_experiment
    # 提取 hartmann_values
    hartmann_values = extract_metric(experiment, "hartman6")
    print(hartmann_values)
    # format like
    # array([-4.65532605e-02, -3.24010025e-01, -1.45934161e-01, -2.14999277e-01,
    #        -3.27793979e-01, -6.82430194e-02, -3.67049044e-01, -6.90476570e-01,
    #       ])

    # 验证其为非空的 numpy array
    assert isinstance(
        hartmann_values, np.ndarray
    ), "hartmann_values is not a numpy array."
    assert hartmann_values.size > 0, "hartmann is empty."


def test_extract_metric_l2norm_values(initialize_experiment):
    experiment, _ = initialize_experiment
    # 提取 l2norm_values
    l2norm_values = extract_metric(experiment, "l2norm")
    print(l2norm_values)
    # format like
    # array([1.53776331, 1.61891927, 0.97083409, 1.39319261, 1.36996788,
    #        1.37336021, 1.35665936, 1.28162942, 1.00141922, 0.93151435,
    # ])

    # 验证其为非空的 numpy array
    assert isinstance(
        l2norm_values, np.ndarray
    ), "l2norm_values is not a numpy array."
    assert l2norm_values.size > 0, "l2norms is empty."
