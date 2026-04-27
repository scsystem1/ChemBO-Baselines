import argparse
import swanlab
from swanlab.plugin.notification import LarkCallback
from src.agents.notes_agent import BaseNotesResponse
from ax import (
    SearchSpace,
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    OptimizationConfig,
    Runner,
    Experiment,
    Objective,
    Models,
)
from src.tasks import (
    ChemistryMetric,
    Hartmann6Metric,
    AckleyMetric,
    LevyMetric,
    RosenbrockMetric,
    LunarLanderMetric,
)

from src.tasks.chemistry.chemistry import ChemistryProblemType

from src.agents import NotesAgent, KGAgent, MilvusAgent
from src.agents.notes_agent import BaseNotesResponse
from pydantic import Field
from typing import List, Dict
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from src.config import Config
import json

config = Config()
# ---------------------------------- Parse arguments ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--exp_config_path', type=str, required=True)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--num_iterations', type=int, required=True)
parser.add_argument('--metric_name', type=str, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument(
    '--reasoner', type=str, required=True, choices=['deepseek', 'qwq', 'kimi', 'local']
)
parser.add_argument('--model_path', type=str)
parser.add_argument('--enable_notes', action='store_true')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument(
    '--ablation_mode',
    type=str,
    default=None,
    choices=['vanilla_first', 'reasoning_first'],
    help='Ablation study mode',
)
args = parser.parse_args()

# Validate arguments
if args.reasoner == 'local' and not args.model_path:
    raise ValueError("--model_path is required when using local reasoner")
if args.reasoner != 'local' and args.model_path:
    print("Warning: --model_path is ignored for API reasoners")


# ---------------------------------- Import metric based on name ----------------------------------
metric_name = args.metric_name
try:
    metric_to_class = {
        "direct_arylation": ChemistryMetric,
        "suzuki": ChemistryMetric,
        "buchwald": ChemistryMetric,
        "CPA": ChemistryMetric,
        "hartmann": Hartmann6Metric,
        "ackley": AckleyMetric,
        "levy": LevyMetric,
        "rosenbrock": RosenbrockMetric,
        "lunar": LunarLanderMetric,
    }
    MetricClass = metric_to_class[metric_name]
except KeyError:
    raise ValueError(f"Could not find metric class for {metric_name}")

# ---------------------------------- load config ----------------------------------
exp_config_path = args.exp_config_path
num_iterations = args.num_iterations
result_dir = args.result_dir
seed = args.seed

# ---------------------------------- SwanLab Init ----------------------------------
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/fd23e392-4357-43ae-a1e2-f6bf56e8cf73",
    secret="edS9WVNfbexWPvkkMt6yie",
)

# ---------------------------------- Define search space ----------------------------------

if metric_name == "ackley":
    parameters = [
        RangeParameter(
            name=f"x{i+1}",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        )
        for i in range(2)  # Adjust range based on problem dimensionality
    ]
elif metric_name == "hartmann":
    parameters = [
        RangeParameter(
            name=f"x{i+1}",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        )
        for i in range(6)
    ]
elif metric_name == "levy":
    parameters = [
        RangeParameter(
            name=f"x{i+1}",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        )
        for i in range(5)
    ]
elif metric_name == "lunar":
    parameters = [
        RangeParameter(
            name="horizontal_position",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="horizontal_velocity",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="angle_limit",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="hover_height",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="angle_error_gain",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="angular_speed_gain",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="vertical_error_gain",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="vertical_speed_gain",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="leg_angle_target",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="leg_vertical_gain",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="hover_threshold",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="angle_threshold",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ]
elif metric_name == "rosenbrock":
    parameters = [
        RangeParameter(
            name=f"x{i+1}",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        )
        for i in range(3)
    ]
elif metric_name == "suzuki":
    parameters = [
        ChoiceParameter(
            name="Electrophile_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "BrC1=CC=C(N=CC=C2)C2=C1",
                "ClC1=CC=C(N=CC=C2)C2=C1",
                "IC1=CC=C(N=CC=C2)C2=C1",
                "O=S(OC1=CC=C(N=CC=C2)C2=C1)(C(F)(F)F)=O",
            ],
        ),
        ChoiceParameter(
            name="Nucleophile_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1[B-](F)(F)F",
                "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O",
                "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B4OC(C)(C)C(C)(C)O4",
            ],
        ),
        ChoiceParameter(
            name="Ligand_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "[c-]1(P(C2=CC=CC=C2)C3=CC=CC=C3)cccc1.[c-]4(P(C5=CC=CC=C5)C6=CC=CC=C6)cccc4.[Fe+2]",
                "CC(C)(C)P(C(C)(C)C)C1=CC=C(N(C)C)C=C1",
                "CC(C)(P(C(C)(C)C)[c-]1cccc1)C.CC(C)(P(C(C)(C)C)[c-]2cccc2)C.[Fe+2]"
                "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
                "CC(P(C(C)(C)C)C(C)(C)C)(C)C",
                "CC1(C)C2=C(OC3=C1C=CC=C3P(C4=CC=CC=C4)C5=CC=CC=C5)C(P(C6=CC=CC=C6)C7=CC=CC=C7)=CC=C2",
                "CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C",
                "CCCCP(C12C[C@@H]3C[C@@H](C[C@H](C2)C3)C1)C45C[C@H]6C[C@@H](C5)C[C@@H](C4)C6",
                "COC1=CC=CC(OC)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
                "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
                "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
            ],
        ),
        ChoiceParameter(
            name="Base_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "[Cs+].[F-]",
                "[K+].[OH-]",
                "[Na+].[OH-]",
                "CC([O-])C.[Li+]",
                "CCN(CC)CC",
                "O=P([O-])([O-])[O-].[K+].[K+].[K+]",
                "OC([O-])=O.[Na+]",
            ],
        ),
        ChoiceParameter(
            name="Solvent_SMILES",
            parameter_type=ParameterType.STRING,
            values=["C1COCC1", "CO", "N#CC", "O=CN(C)C"],
        ),
    ]
elif metric_name == "buchwald":
    parameters = [
        ChoiceParameter(
            name="Ligand",
            parameter_type=ParameterType.STRING,
            values=[
                "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC",
                "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC",
                "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2",
                "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
            ],
        ),
        ChoiceParameter(
            name="Additive",
            parameter_type=ParameterType.STRING,
            values=[
                "C1(C2=CC=CC=C2)=CC=NO1",
                "C1(C2=CC=CC=C2)=CON=C1",
                "C1(C2=CC=CC=C2)=NOC=C1",
                "C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=CC=NO1",
                "C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=NOC=C1",
                "C12=C(C=CC=C2)ON=C1",
                "C12=CON=C1C=CC=C2",
                "CC1=C(C(OCC)=O)C=NO1",
                "CC1=CC(C(OCC)=O)=NO1",
                "CC1=CC(C(OCC)=O)=NO1",
                "CC1=CC(N2C=CC=C2)=NO1",
                "CC1=CC=NO1",
                "CC1=NOC(C(OCC)=O)=C1",
                "CC1=NOC(C2=CC=CC=C2)=C1",
                "CC1=NOC=C1",
                "CCOC(C1=CON=C1)=O",
                "CCOC(C1=NOC=C1)=O",
                "COC1=NOC(C(OCC)=O)=C1",
                "FC(C=CC=C1F)=C1C2=CC=NO2",
                "O=C(OC)C1=CC=NO1",
                "O=C(OC)C1=NOC(C2=CC=CO2)=C1",
                "O=C(OC)C1=NOC(C2=CC=CS2)=C1",
            ],
        ),
        ChoiceParameter(
            name="Base",
            parameter_type=ParameterType.STRING,
            values=[
                "CC(C)(C)/N=C(N(C)C)/N(C)C",
                "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC",
                "CN1CCCN2C1=NCCC2",
            ],
        ),
        ChoiceParameter(
            name="Aryl halide",
            parameter_type=ParameterType.STRING,
            values=[
                "BrC1=CC=C(C(F)(F)F)C=C1",
                "BrC1=CC=C(CC)C=C1",
                "BrC1=CC=C(OC)C=C1",
                "BrC1=CN=CC=C1",
                "BrC1=NC=CC=C1",
                "ClC1=CC=C(C(F)(F)F)C=C1",
                "ClC1=CC=C(CC)C=C1",
                "ClC1=CC=C(OC)C=C1",
                "ClC1=CN=CC=C1",
                "ClC1=NC=CC=C1",
                "IC1=CC=C(C(F)(F)F)C=C1",
                "IC1=CC=C(CC)C=C1",
                "IC1=CC=C(OC)C=C1",
                "IC1=CN=CC=C1",
                "IC1=NC=CC=C1",
            ],
        ),
    ]
elif metric_name == "CPA":
    parameters = [
        ChoiceParameter(
            name="Catalyst",
            parameter_type=ParameterType.STRING,
            values=[
                'O=P1(O)OC2=C(C3=CC=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(F)C=C(OC)C=C3F)C=C4C(C=CC=C4)=[C@]2[C@]5=C(O1)C(C6=C(F)C=C(OC)C=C6F)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(C)C=C(C)C=C3C)C=C4C(C=CC=C4)=C2C5=C(O1)[C@@]([C@@]6=C(C)C=C(C)C=C6C)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(C)=C(OC(C)C)C(C)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C)=C(OC(C)C)C(C)=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC=C(S(F)(F)(F)(F)F)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(S(F)(F)(F)(F)F)C=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C5=CC(C(F)(F)F)=CC(C(F)(F)F)=C5)=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC(C9=CC(C(F)(F)F)=CC(C(F)(F)F)=C9)=CC(C%10=CC(C(F)(F)F)=CC(C(F)(F)F)=C%10)=C8)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C(CC3=CC=C(OC)C=C3)C=C4C(CCCC4)=C2C5=C(O1)C(CC6=CC=C(OC)C=C6)=CC7=C5CCCC7',
                'O=P1(O)OC2=C(CC3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C=C4C(CCCC4)=C2C5=C(O1)C(CC6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)=CC7=C5CCCC7',
                'O=P1(O)OC2=C(CC3=C(C=CC=C4)C4=CC5=C3C=CC=C5)C=C6C(C=CC=C6)=C2C7=C(O1)C(CC8=C(C=CC=C9)C9=CC%10=C8C=CC=C%10)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C(CC3=CC=C(C(F)(F)F)C=C3C(F)(F)F)C=C4C(C=CC=C4)=C2C5=C(O1)C(CC6=C(C(F)(F)F)C=C(C(F)(F)F)C=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C([Si](C3=CC=CC=C3)(C)C4=CC=CC=C4)C=C5C(C=CC=C5)=C2C6=C(O1)C([Si](C7=CC=CC=C7)(C8=CC=CC=C8)C)=CC9=C6C=CC=C9',
                'O=P1(O)OC2=C(C3=CC(C4=CC=C(OC)C=C4)=CC(C5=CC=C(OC)C=C5)=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC(C9=CC=C(OC)C=C9)=CC(C%10=CC=C(OC)C=C%10)=C8)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C(Br)C=C3C(C=CC=C3)=C2C4=C(O1)C(Br)=CC5=CC=CC=C54',
                'O=P1(O)OC2=C(Br)C=C3C(CCCC3)=C2C4=C(O1)C(Br)=CC5=C4CCCC5',
                'O=P1(O)OC2=C([Si](C3=CC=CC=C3)(C4=CC=CC=C4)C5=CC=CC=C5)C=C6C(C=CC=C6)=C2C7=C(O1)C([Si](C8=CC=CC=C8)(C9=CC=CC=C9)C%10=CC=CC=C%10)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C([Si](C3=CC=CC=C3)(C4=CC=CC=C4)C5=CC=CC=C5)C=C6C(CCCC6)=C2C7=C(O1)C([Si](C8=CC=CC=C8)(C9=CC=CC=C9)C%10=CC=CC=C%10)=CC%11=C7CCCC%11',
                'O=P1(O)OC2=C([Si](C3=CC=C(C(C)(C)C)C=C3)(C4=CC=C(C(C)(C)C)C=C4)C5=CC=C(C(C)(C)C)C=C5)C=C6C(CCCC6)=C2C7=C(O1)C([Si](C8=CC=C(C(C)(C)C)C=C8)(C9=CC=C(C(C)(C)C)C=C9)C%10=CC=C(C(C)(C)C)C=C%10)=CC%11=C7CCCC%11',
                'O=P1(O)OC2=C(C3=CC(C4=C(C)C=C(C)C=C4C)=CC(C5=C(C)C=C(C)C=C5C)=C3)C=C6C(CCCC6)=C2C7=C(O1)C(C8=CC(C9=C(C)C=C(C)C=C9C)=CC(C%10=C(C)C=C(C)C=C%10C)=C8)=CC%11=C7CCCC%11',
                'O=P1(O)OC2=C(C3=C(C(C)C)C=C(C4=CC=C(C(C)(C)C)C=C4)C=C3C(C)C)C=C5C(C=CC=C5)=[C@]2[C@]6=C(O1)C(C7=C(C(C)C)C=C(C8=CC=C(C(C)(C)C)C=C8)C=C7C(C)C)=CC9=C6C=CC=C9',
                'O=P1(O)OC2=C(CC)C=C3C(CCCC3)=C2C4=C(O1)C(CC)=CC5=C4CCCC5',
                'O=P1(O)OC2=[C@]([C@]3=C(Cl)C=C(Cl)C=C3Cl)C=C4C(CCCC4)=[C@]2[C@]5=C(O1)C(C6=C(Cl)C=C(Cl)C=C6Cl)=CC7=C5CCCC7',
                'O=P1(O)OC2=C(C3=CC=C(OC)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(OC)C=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(OCC)C=CC(C)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C)=CC=C6OCC)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(COC)=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC(COC)=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(COC)=CC=C3)C=C4C(CCCC4)=C2C5=C(O1)C(C6=CC=CC(COC)=C6)=CC7=C5CCCC7',
                'O=P1(O)OC2=C(C3=CC=C(C)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(C)C=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(OC(F)(F)F)C=CC=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=CC=C6OC(F)(F)F)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(C=CC=C4)C4=CC5=C3C=CC=C5)C=C6C(C=CC=C6)=C2C7=C(O1)[C@@]([C@@]8=C(C=CC=C9)C9=CC%10=C8C=CC=C%10)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C(C3=CC=C(C(C)(C)C)C=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC=C(C(C)(C)C)C=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(C=CC4=CC=CC(C=C5)=C46)C6=C5C=C3)C=C7C(C=CC=C7)=C2C8=C(O1)C(C9=CC=C(C=C%10)C%11=C9C=CC%12=CC=CC%10=C%11%12)=CC%13=C8C=CC=C%13',
                'O=P1(O)OC2=C(C3=C(C4=CC(C=CC=C5)=C5C=C4)C=CC=C3)C=C6C(CCCC6)=C2C7=C(O1)C(C8=CC=CC=C8C9=CC=C(C=CC=C%10)C%10=C9)=CC%11=C7CCCC%11',
                'O=P1(O)OC2=C(C3=CC(C4=CC(C=CC=C5)=C5C=C4)=CC=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC=CC(C9=CC=C(C=CC=C%10)C%10=C9)=C8)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C(C3=CC=C(C4=CC=C(C=CC=C5)C5=C4)C=C3)C=C6C(C=CC=C6)=C2C7=C(O1)C(C8=CC=C(C9=CC(C=CC=C%10)=C%10C=C9)C=C8)=CC%11=C7C=CC=C%11',
                'O=P1(O)OC2=C(C3=CC=C(C4CCCCC4)C=C3)C=C5C(CCCC5)=C2C6=C(O1)C(C7=CC=C(C8CCCCC8)C=C7)=CC9=C6CCCC9',
                'O=P1(O)OC2=C(C3=C(OC)C=CC=C3OC)C=C4C(C=CC=C4)=C2C5=C(O1)[C@@]([C@@]6=C(OC)C=CC=C6OC)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C=C4C(CCCC4)=C2C5=C(O1)C(C6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)=CC7=C5CCCC7',
                'O=P1(O)OC2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C=C4C(C=CC=C4)=C2C5=C(O1)[C@@]([C@@]6=C(C(C)C)C=C(C(C)C)C=C6C(C)C)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=C(C4CCCCC4)C=C(C5CCCCC5)C=C3C6CCCCC6)C=C7C(C=CC=C7)=C2C8=C(O1)[C@@]([C@@]9=C(C%10CCCCC%10)C=C(C%11CCCCC%11)C=C9C%12CCCCC%12)=CC%13=C8C=CC=C%13',
                'O=P1(O)OC2=C(C3=CC(C(C)(C)C)=CC(C(C)(C)C)=C3)C=C4C(C=CC=C4)=C2C5=C(O1)C(C6=CC(C(C)(C)C)=CC(C(C)(C)C)=C6)=CC7=C5C=CC=C7',
                'O=P1(O)OC2=C(C3=CC(C(C)(C)C)=CC(C(C)(C)C)=C3)C=C4C(CCCC4)=C2C5=C(O1)C(C6=CC(C(C)(C)C)=CC(C(C)(C)C)=C6)=CC7=C5CCCC7',
                'O=P1(O)OC2=C(C3=CC=C(C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)C=C3)C=C5C(C=CC=C5)=C2C6=C(O1)C(C7=CC=C(C8=CC(C(F)(F)F)=CC(C(F)(F)F)=C8)C=C7)=CC9=C6C=CC=C9',
                'O=P1(O)OC2=C(C3=C(C=CC=C4)C4=C(C5=CC(C=CC=C6)=C6C=C5)C7=C3C=CC=C7)C=C8C(C=CC=C8)=[C@]2[C@]9=C(O1)C(C%10=C(C=CC=C%11)C%11=C(C%12=CC=C(C=CC=C%13)C%13=C%12)C%14=C%10C=CC=C%14)=CC%15=C9C=CC=C%15',
            ],
        ),
        ChoiceParameter(
            name="Imine",
            parameter_type=ParameterType.STRING,
            values=[
                "O=C(C1=CC=CC=C1)/N=C/C2=CC=C(C(F)(F)F)C=C2",
                "O=C(C1=CC=CC=C1)/N=C/C2=CC=C(Cl)C=C2Cl",
                "O=C(C1=CC=CC=C1)/N=C/C2=CC=C(OC)C=C2",
                "O=C(C1=CC=CC=C1)/N=C/C2=CC=CC=C2",
                "O=C(C1=CC=CC=C1)/N=C/C2=CC=CC3=C2C=CC=C3",
            ],
        ),
        ChoiceParameter(
            name="Thiol",
            parameter_type=ParameterType.STRING,
            values=[
                "CCS",
                "SC1=CC=C(OC)C=C1",
                "SC1=CC=CC=C1",
                "SC1=CC=CC=C1C",
                "SC1CCCCC1",
            ],
        ),
    ]
elif metric_name == "direct_arylation":
    parameters = [
        ChoiceParameter(
            name="Base_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "O=C([O-])C(C)(C)C.[Cs+]",
                "O=C([O-])C(C)(C)C.[K+]",
                "O=C([O-])C.[Cs+]",
                "O=C([O-])C.[K+]",
            ],
        ),
        ChoiceParameter(
            name="Ligand_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1",
                "CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
                "CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)=CC=C2OC",
                "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
                "CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C",
                "CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
                "CP(C)C1=CC=CC=C1",
                "CP(C1=CC=CC=C1)C2=CC=CC=C2",
                "FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1",
                "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
                "P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3",
                "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
            ],
        ),
        ChoiceParameter(
            name="Solvent_SMILES",
            parameter_type=ParameterType.STRING,
            values=[
                "CC(N(C)C)=O",
                "CC1=CC=C(C)C=C1",
                "CCCC#N",
                "CCCCOC(C)=O",
            ],
        ),
        ChoiceParameter(
            name="Concentration",
            parameter_type=ParameterType.FLOAT,
            values=[
                0.057,
                0.1,
                0.153,
            ],
        ),
        ChoiceParameter(
            name="Temp_C",
            parameter_type=ParameterType.INT,
            values=[90, 105, 120],
        ),
    ]
else:
    raise ValueError(
        f"Unknown metric name: {metric_name}. Please implement parameters for this metric."
    )


search_space = SearchSpace(parameters=parameters)

# ---------------------------------- Create Optimization Config ----------------------------------
if metric_name == "direct_arylation":
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=ChemistryMetric(
                name=metric_name,
                noiseless=True,
                problem_type=ChemistryProblemType.DIRECT_ARYLATION,
            )
        )
    )
elif metric_name == "buchwald":
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=ChemistryMetric(
                name=metric_name,
                problem_type=ChemistryProblemType.Buchwald_Hartwig,
                noiseless=True,
            )
        )
    )
elif metric_name == "CPA":
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=ChemistryMetric(
                name=metric_name,
                problem_type=ChemistryProblemType.CPA,
                noiseless=True,
            )
        )
    )
elif metric_name == "lunar":
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=MetricClass(
                name=metric_name,
            )
        )
    )
else:
    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=MetricClass(
                name=metric_name,
                noiseless=True,
            )
        )
    )


# ---------------------------------- Define a Runner ----------------------------------
class Runner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


# Select reasoner based on input
if args.reasoner == 'deepseek':
    from src.bo.reasoner.deepseek import DSReasoner

    reasoner = DSReasoner(
        exp_config_path=exp_config_path, result_dir=result_dir
    )
if args.reasoner == 'qwq':
    from src.bo.reasoner.qwq import QWQReasoner

    reasoner = QWQReasoner(
        exp_config_path=exp_config_path, result_dir=result_dir
    )
if args.reasoner == 'local':
    from src.bo.reasoner.local_lm import LocalLMReasoner

    reasoner = LocalLMReasoner(
        exp_config_path=exp_config_path,
        result_dir=result_dir,
        model_path=args.model_path,
    )
if args.reasoner == 'kimi':
    from src.bo.reasoner.kimi import KimiReasoner

    reasoner = KimiReasoner(
        exp_config_path=exp_config_path, result_dir=result_dir
    )

swanlab.init(
    project=metric_name,
    config={
        "exp_config_path": exp_config_path,
        "result_dir": result_dir,
        "num_iterations": num_iterations,
        "model": args.model_type,
        "seed": seed,
        "enable_notes": args.enable_notes,
    },
    description="vanilla_first",
    callbacks=[lark_callback],
)
# ---------------------------------- Set random seeds for reproducibility ----------------------------------
import random
import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------------- Create Experiment ----------------------------------
exp = Experiment(
    name=metric_name,
    search_space=search_space,
    optimization_config=optimization_config,
    runner=Runner(),
)

# ---------------------------------- Main experiment loop ----------------------------------
from src.bo.models import BOModel

bo_model = BOModel(exp)

if args.enable_notes:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        api_key=config.QWQ_API_KEY,
        url=config.QWQ_API_BASE,
        model_type=config.NOTES_AGENT,
        model_config_dict={"temperature": 0, "max_tokens": 2048},
    )

    kg_agent = KGAgent(model)
    milvus_agent = MilvusAgent(collection_name="test")

    notes_agent = NotesAgent(
        model=model, kg_agent=kg_agent, milvus_agent=milvus_agent
    )

    client = reasoner.client
    # ---------------------------------- extract notes from experiment compass ----------------------------------
    with open(exp_config_path, "r") as f:
        experiment_compass = json.load(f)

    experiment_str = json.dumps(
        experiment_compass, indent=2, ensure_ascii=False
    )
    experiment_info = notes_agent.extract_compass_notes(
        experiment_data=experiment_str,
        save_schema=BaseNotesResponse,
    )

ablation_mode = args.ablation_mode

if ablation_mode == 'vanilla_first':
    # Vanilla BO + Reasoning BO
    # Sobol * 4 + Reasoning BO * 27
    sobol = Models.SOBOL(exp.search_space)
    for i in range(4):
        generator_run = sobol.gen(3)
        trial = exp.new_batch_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()
        max_value = trial.fetch_data().df['mean'].max()
        mean_value = trial.fetch_data().df['mean'].mean()
        min_value = trial.fetch_data().df['mean'].min()

        swanlab.log(
            {
                "max_value": float(max_value),
                "mean_value": float(mean_value),
                "min_value": float(min_value),
            },
            step=i + 1,
        )
    num_iterations = 30
    overview = reasoner.generate_overview()
    print(overview)

    # 2. Initial sampling
    insight_first_round = reasoner.initial_sampling()

    # 3. First round experiment
    candidates_array = reasoner.optimization_first_round(insight_first_round)
    trial = reasoner.run_bo_experiment(exp, candidates_array)
    reasoner._save_experiment_data(exp, trial)

    for i in range(4, num_iterations + 4):
        if args.enable_notes:
            keywords = reasoner.get_keywords()
            print(f"Keywords:{keywords}")
            retrieved_context = ''
            if keywords:
                results = notes_agent.query_notes(
                    query=keywords, top_k=3, similarity_threshold=0.7
                )
                retrieved_context = notes_agent.format_retrieved_notes(results)
            print(f"retrieved context: \n{retrieved_context}")

        candidates_array = reasoner.optimization_loop(
            experiment=exp, trial=trial, bo_model=bo_model, n=5
        )
        if args.enable_notes:
            for msg in client.messages:
                if msg.get("role") == "think":
                    reasoning_data = msg.get("content")
            knowledge = notes_agent.extract_reasoning_notes(
                reasoning_data=reasoning_data,
                save_schema=BaseNotesResponse,
            )
        trial = reasoner.run_bo_experiment(exp, candidates_array)

        max_value = trial.fetch_data().df['mean'].max()
        mean_value = trial.fetch_data().df['mean'].mean()
        min_value = trial.fetch_data().df['mean'].min()

        swanlab.log(
            {
                "max_value": float(max_value),
                "mean_value": float(mean_value),
                "min_value": float(min_value),
            },
            step=i + 1,
        )

elif ablation_mode == 'reasoning_first':
    # Reasoning BO + Vanilla BO
    # initial sample + reasoning bo * 3 + Vanilla BO
    # 1. Generate overview
    overview = reasoner.generate_overview()
    print(overview)

    # 2. Initial sampling
    insight_first_round = reasoner.initial_sampling()

    # ---------------------------------- extract notes from cot ----------------------------------
    if args.enable_notes:
        for msg in client.messages:
            if msg.get("role") == "think":
                reasoning_data = msg.get("content")
        knowledge = notes_agent.extract_reasoning_notes(
            reasoning_data=reasoning_data,
            save_schema=BaseNotesResponse,
        )
        print(f"Extracted Notes from reasoning data:{knowledge}")

    # 3. First round experiment
    candidates_array = reasoner.optimization_first_round(insight_first_round)
    trial = reasoner.run_bo_experiment(exp, candidates_array)
    reasoner._save_experiment_data(exp, trial)

    # 4. Optimization loop
    for i in range(3):
        if args.enable_notes:
            keywords = reasoner.get_keywords()
            print(f"Keywords:{keywords}")
            retrieved_context = ''
            if keywords:
                results = notes_agent.query_notes(
                    query=keywords, top_k=3, similarity_threshold=0.7
                )
                retrieved_context = notes_agent.format_retrieved_notes(results)
            print(f"retrieved context: \n{retrieved_context}")

        candidates_array = reasoner.optimization_loop(
            experiment=exp, trial=trial, bo_model=bo_model, n=5
        )
        if args.enable_notes:
            for msg in client.messages:
                if msg.get("role") == "think":
                    reasoning_data = msg.get("content")
            knowledge = notes_agent.extract_reasoning_notes(
                reasoning_data=reasoning_data,
                save_schema=BaseNotesResponse,
            )
        trial = reasoner.run_bo_experiment(exp, candidates_array)

        max_value = trial.fetch_data().df['mean'].max()
        mean_value = trial.fetch_data().df['mean'].mean()
        min_value = trial.fetch_data().df['mean'].min()

        swanlab.log(
            {
                "max_value": float(max_value),
                "mean_value": float(mean_value),
                "min_value": float(min_value),
            },
            step=i + 1,
        )
        num_iterations = 30

    for i in range(3, num_iterations + 3):
        generator_run = bo_model.gen(n=3)
        candidates_array = [arm.parameters for arm in generator_run.arms]
        trial = reasoner.run_bo_experiment(exp, candidates_array)

        max_value = trial.fetch_data().df['mean'].max()
        mean_value = trial.fetch_data().df['mean'].mean()
        min_value = trial.fetch_data().df['mean'].min()

        swanlab.log(
            {
                "max_value": float(max_value),
                "mean_value": float(mean_value),
                "min_value": float(min_value),
            },
            step=i + 1,
        )
else:
    # 原始模式
    # 1. Generate overview
    overview = reasoner.generate_overview()
    print(overview)

    # 2. Initial sampling
    insight_first_round = reasoner.initial_sampling()

    # ---------------------------------- extract notes from cot ----------------------------------
    if args.enable_notes:
        for msg in client.messages:
            if msg.get("role") == "think":
                reasoning_data = msg.get("content")
        knowledge = notes_agent.extract_reasoning_notes(
            reasoning_data=reasoning_data,
            save_schema=BaseNotesResponse,
        )
        print(f"Extracted Notes from reasoning data:{knowledge}")

    # 3. First round experiment
    candidates_array = reasoner.optimization_first_round(insight_first_round)
    trial = reasoner.run_bo_experiment(exp, candidates_array)
    reasoner._save_experiment_data(exp, trial)

    # 4. Optimization loop
    for i in range(num_iterations):
        if args.enable_notes:
            keywords = reasoner.get_keywords()
            print(f"Keywords:{keywords}")
            retrieved_context = ''
            if keywords:
                results = notes_agent.query_notes(
                    query=keywords, top_k=3, similarity_threshold=0.7
                )
                retrieved_context = notes_agent.format_retrieved_notes(results)
            print(f"retrieved context: \n{retrieved_context}")

        candidates_array = reasoner.optimization_loop(
            experiment=exp, trial=trial, bo_model=bo_model, n=5
        )
        if args.enable_notes:
            for msg in client.messages:
                if msg.get("role") == "think":
                    reasoning_data = msg.get("content")
            knowledge = notes_agent.extract_reasoning_notes(
                reasoning_data=reasoning_data,
                save_schema=BaseNotesResponse,
            )
        trial = reasoner.run_bo_experiment(exp, candidates_array)

        max_value = trial.fetch_data().df['mean'].max()
        mean_value = trial.fetch_data().df['mean'].mean()
        min_value = trial.fetch_data().df['mean'].min()

        swanlab.log(
            {
                "max_value": float(max_value),
                "mean_value": float(mean_value),
                "min_value": float(min_value),
            },
            step=i + 1,
        )

# 5. Generate experiment analysis
reasoner.generate_experiment_analysis()

# ---------------------------------- Results analysis ----------------------------------
from src.utils.metric import extract_metric

max_results = extract_metric(exp=exp, metric_name=metric_name, mode='max')
print("max_results:", max_results)
mean_results = extract_metric(exp=exp, metric_name=metric_name, mode='mean')
print("mean_results:", mean_results)
min_results = extract_metric(exp=exp, metric_name=metric_name, mode='min')
print("min_results:", min_results)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create figure with three subplots
plt.figure(figsize=(18, 6))

# Subplot 1: Max results
plt.subplot(1, 3, 1)
plt.plot(
    range(1, len(max_results) + 1),
    max_results,
    marker='o',
    linestyle='-',
    color='#1f77b4',
    label='Max reward',
)
plt.xlabel('Trial Number')
plt.ylabel('Yield (%)')
plt.title('Maximum reward Progress')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Mean results
plt.subplot(1, 3, 2)
plt.plot(
    range(1, len(mean_results) + 1),
    mean_results,
    marker='s',
    linestyle='--',
    color='#ff7f0e',
    label='Mean Reward',
)
plt.xlabel('Trial Number')
plt.ylabel('Yield (%)')
plt.title('Average Reward Progress')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3: Min results
plt.subplot(1, 3, 3)
plt.plot(
    range(1, len(min_results) + 1),
    min_results,
    marker='^',
    linestyle=':',
    color='#2ca02c',
    label='Min Reward',
)
plt.xlabel('Trial Number')
plt.ylabel('Yield (%)')
plt.title('Minimum Reward Progress')
plt.grid(True, alpha=0.3)
plt.legend()

# Log the plot
swanlab.log({"Value": swanlab.Image(plt)})

swanlab.finish()
# plt.show()
