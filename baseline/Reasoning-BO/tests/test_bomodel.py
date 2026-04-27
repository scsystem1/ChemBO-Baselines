import pytest
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
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from src.bo.models import DSReasoner

# ---------------------------------- define a runner ----------------------------------


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


@pytest.fixture
def experiment():
    # ---------------------------------- create search space ----------------------------------
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

    # ---------------------------------- create optimization config ----------------------------------
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
    # ---------------------------------- create experiment ----------------------------------
    exp = Experiment(
        name="test_hartmann",
        search_space=hartmann_search_space,
        optimization_config=optimization_config,
        runner=MyRunner(),
    )
    return exp


class Test_BOModel:
    def test_bo_model(self, experiment):
        bo_model = BOModel(experiment)
        bo_model.hot_start(experiment)
        for i in range(2):
            generator_run = bo_model.gen(n=3)
            print(generator_run.arms)
            # format like [Arm(parameters={'x1': -5.0, 'x2': 0.0}), Arm(parameters={'x1': -5.0, 'x2': 15.0}), Arm(parameters={'x1': 10.0, 'x2': 0.0})]
            candidates = [arm.parameters for arm in generator_run.arms]
            print(candidates)
            # format like [{'x1': -4, 'x2': 0.0}, {'x1': 10.0, 'x2': 4}, {'x1': -5.0, 'x2': 15.0}]
            # TODO
            # add llm filtering logic
            # llm = R1Reasoner()
            # filtered_arms = llm.recommend(candidates)
            # print(filtered_arms)
            # format like [{'x1': -4, 'x2': 0.0}]
            filtered_candidates = [
                Arm(parameters=params) for params in candidates
            ]
            filtered_generator_run = GeneratorRun(arms=filtered_candidates)
            trial = experiment.new_batch_trial(
                generator_run=filtered_generator_run
            )
            trial.run()
            trial.mark_completed()
            assert candidates is not None
            assert generator_run.arms is not None
            assert filtered_candidates is not None
        assert bo_model.model_bridge is not None


class TestDSReasoner:
    def test_generate_overview(self):
        file_path = "Reasoning-BO/src/config/hartmann6_config.json"
        ds_reasoner = DSReasoner(file_path)
        overview = ds_reasoner.generate_overview()
        print(overview)
        assert overview is not None
