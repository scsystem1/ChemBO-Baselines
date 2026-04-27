import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

from utils import *
from utils import run_llm_survey

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run LLM Survey")
parser.add_argument("--questions", required=True, help="Path to the questions file (.npy or .pkl)")
parser.add_argument("--exp_database", required=True, help="Path to the experimental database CSV file")
parser.add_argument("--res_dir", required=True, help="Directory to save results")
parser.add_argument("--llm_result", required=True, help="Name of the output CSV file")
parser.add_argument("--model_type", type=int, default=0, help="Type of LLM model to use")
args = parser.parse_args()

# Load questions from secure NumPy format (no code execution risk)
try:
    questions = np.load(args.questions, allow_pickle=False)
except Exception as e:
    raise RuntimeError(
        f"Failed to load questions file: {str(e)}\n"
        f"Please provide a .npy file or convert using: python convert_pickle_to_npz.py"
    )

questions = questions[:2000]  # Use 2000 questions
    
# Load experimental database
exp_database = pd.read_csv(args.exp_database)
task_reaction_DA = config.get("TASK_REACTION_DA")

# Context can be None or come from Bayesian Optimization
df_context = exp_database.head()
print('Context dataframe\n', df_context)

# Run experiment
df_results, accuracy, num_questions = run_llm_survey_parallel(
    index=0,
    questions=questions,
    exp_database=exp_database,
    task_reaction=task_reaction_DA,
    df_context=df_context,
    res_dir=args.res_dir,
    model_type=args.model_type,
    max_workers=5
)

print('accuracy', accuracy, 'num_questions', num_questions)
df_results.to_csv(f"{args.llm_result}", index=False)