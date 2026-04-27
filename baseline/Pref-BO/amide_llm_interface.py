"""
LLM interface adapted for amide coupling reactions using native boto3.
Based on the existing LLM/utils.py but modified for amide coupling dataset format
and using native boto3 client like the working example.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import warnings
import time

# Add LLM directory to path to import original utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'LLM'))

import config
from kimi_client import call_kimi_chat, parse_jsonish_response

warnings.filterwarnings('ignore')


def call_bedrock_claude(user_prompt=None, model_type=0):
    """
    Call Claude via AWS Bedrock using native boto3 client.
    model_type: 0=Claude 3.5 Sonnet, 1=Claude 3 Sonnet, 2=Claude 3 Haiku
    """
    del model_type
    return call_kimi_chat(user_prompt)


def get_amide_prompt(q_idx, questions, exp_database, df_context=None):
    """
    Generate prompt for amide coupling reactions.
    Adapted for amide coupling dataset with actual column names.
    """
    idx_a = questions[q_idx][0]
    idx_b = questions[q_idx][1]
    
    # Define the general amide coupling reaction
    task_reaction = config.get("TASK_REACTION_AMIDE")
    
    if df_context is None:
        query = "You can predict reaction yield for amide coupling reactions."        
    else:
        query = "You can predict reaction yield for amide coupling reactions using the following examples and your chemistry knowledge.\n"
        query += "Examples:\n" 
        query += "Entry, Sub1_SMILES, Sub2_SMILES, Activation_ID, Additive_ID, Base_ID, Solvent_ID, Yield\n"
        for i in range(min(df_context.shape[0], 5)):  # Limit to 5 examples
            query += f"""{df_context.iloc[i]['sub_1_smiles']}, {df_context.iloc[i]['sub_2_smiles']}, {df_context.iloc[i]['Activation_ID']}, {df_context.iloc[i]['Additive_ID']}, {df_context.iloc[i]['Base_ID']}, {df_context.iloc[i]['solvent_id']}, {df_context.iloc[i]['yield']}\n"""
    
    query += f"""\nTask: For the following amide coupling reaction predict which experiment setup leads to a higher yield and output the response (A or B) and the reasoning in JSON format.
    Task reaction: {task_reaction}
    For any amide coupling reaction, an activation reagent, additive, base, and solvent are present.

    Setup A:
    Substrate 1 (Amine): {exp_database.iloc[idx_a]['sub_1_smiles']}
    Substrate 2 (Acid): {exp_database.iloc[idx_a]['sub_2_smiles']}
    Activation Reagent ID: {exp_database.iloc[idx_a]['Activation_ID']}
    Additive ID: {exp_database.iloc[idx_a]['Additive_ID']}
    Base ID: {exp_database.iloc[idx_a]['Base_ID']}
    Solvent ID: {exp_database.iloc[idx_a]['solvent_id']}
    
    Setup B:
    Substrate 1 (Amine): {exp_database.iloc[idx_b]['sub_1_smiles']}
    Substrate 2 (Acid): {exp_database.iloc[idx_b]['sub_2_smiles']}
    Activation Reagent ID: {exp_database.iloc[idx_b]['Activation_ID']}
    Additive ID: {exp_database.iloc[idx_b]['Additive_ID']}
    Base ID: {exp_database.iloc[idx_b]['Base_ID']}
    Solvent ID: {exp_database.iloc[idx_b]['solvent_id']}
    
    Output: Reaction setup (A or B) with higher yield and the reasoning in JSON object with 'Setup' and 'reasoning' keys
    """
    
    return query


def process_single_amide_question(args):
    """
    Process a single amide coupling preference question using native boto3.
    """
    index, questions, exp_database, df_context, res_dir, entry_sleep, time_out_sleep = args
    
    max_retries = 5
    for attempt in range(max_retries):
        prompt_molecule = get_amide_prompt(index, questions, exp_database, df_context)
        
        try:
            response_first = call_bedrock_claude(prompt_molecule)
        except:
            time.sleep(60)
            continue

        yield_a = exp_database.iloc[questions[index][0]]['yield']
        yield_b = exp_database.iloc[questions[index][1]]['yield']
        true_setup = 'A' if yield_a >= yield_b else 'B'

        query_prompt = f"""
        From the following return only the predicted yield in Json format with 'Setup' and 'reasoning' columns.

        Input: {response_first}
        Output template:
        {{
          "Setup":  ,
          "reasoning": 
        }}
        """

        try:
            response = call_bedrock_claude(query_prompt, model_type=2)  # Use Haiku for follow-up
        except:
            time.sleep(time_out_sleep)
            continue

        if "\n\n" in response:
            ans = response.split("\n\n")[1]
        else:
            try:
                ans = response.split("json\n")[1]
            except:
                ans = response

        try:
            generated_data = parse_jsonish_response(ans)
        except:
            continue

        try:
            dummy = generated_data
        except:
            continue

        try:
            df_generated = pd.DataFrame({
                'question': index, 
                'pred_setup': generated_data['Setup'], 
                'true_setup': true_setup, 
                'reasoning': generated_data['reasoning']
            }, index=[0])
        except:
            continue

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        df_generated.to_csv(f'{res_dir}/df_{index}.csv', index=False)
        time.sleep(entry_sleep)
        
        return df_generated

    # If all retries fail, return None
    return None


def run_amide_llm_survey_parallel(questions, exp_database, df_context=None, res_dir='./results_survey/amide_r1', 
                                 entry_sleep=1, time_out_sleep=60, max_workers=5, max_questions=None):
    """
    Run LLM survey for amide coupling reactions in parallel using native boto3.
    
    Args:
        questions: numpy array of shape (n_questions, 2) with pairs of experiment indices
        exp_database: pandas DataFrame with amide coupling experiments
        df_context: optional context experiments to provide as examples
        res_dir: directory to save results
        entry_sleep: sleep time between questions
        time_out_sleep: timeout sleep for retries
        max_workers: number of parallel workers
        max_questions: maximum number of questions to process (for testing)
    
    Returns:
        df_result: DataFrame with results
        accuracy: accuracy score
        num_questions: number of questions processed
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Limit questions if specified (for testing)
    if max_questions is not None:
        questions = questions[:max_questions]

    args_list = [(i, questions, exp_database, df_context, res_dir, entry_sleep, time_out_sleep) 
                 for i in range(questions.shape[0])]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_single_amide_question, args): args[0] for args in args_list}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(args_list), desc="Processing"):
            index = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f'Question {index} generated an exception: {exc}')

    if len(results) == 0:
        print("No valid results obtained")
        return pd.DataFrame(), 0.0, 0

    df_result = pd.concat(results, ignore_index=True)
    correct = sum(df_result['pred_setup'] == df_result['true_setup'])
    num = len(df_result)
    acc = correct / num if num > 0 else 0

    return df_result, acc, num


def run_amide_llm_survey(questions, exp_database, df_context=None, res_dir='./results_survey/amide_r1', 
                        entry_sleep=1, time_out_sleep=60, max_questions=None):
    """
    Run LLM survey for amide coupling reactions sequentially using native boto3.
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    # Limit questions if specified (for testing)
    if max_questions is not None:
        questions = questions[:max_questions]
    
    progress_bar = tqdm(total=questions.shape[0], desc="Processing")
    index = 0
    
    while index < questions.shape[0]:
        prompt_molecule = get_amide_prompt(index, questions, exp_database, df_context)
        
        try:
            response_first = call_bedrock_claude(prompt_molecule)
        except:
            time.sleep(60)
            continue

        yield_a = exp_database.iloc[questions[index][0]]['yield']
        yield_b = exp_database.iloc[questions[index][1]]['yield']
        true_setup = 'A' if yield_a >= yield_b else 'B'

        query_prompt = f"""
        From the following return only the predicted yield in Json format with 'Setup' and 'reasoning' columns.

        Input: {response_first}"""
        query_prompt += """
        Output template:
        {
      "Setup":  ,
      "reasoning": 
        }
        """

        try:
            response = call_bedrock_claude(query_prompt, model_type=2)
        except:
            time.sleep(time_out_sleep)
            continue

        if "\n\n" in response:
            ans = response.split("\n\n")[1]
        else:
            try:
                ans = response.split("json\n")[1]
            except:
                ans = response

        try:
            generated_data = parse_jsonish_response(ans)
        except:
            continue

        try:
            dummy = generated_data
        except:
            continue

        try:
            df_generated = pd.DataFrame({
                'question': index, 
                'pred_setup': generated_data['Setup'], 
                'true_setup': true_setup, 
                'reasoning': generated_data['reasoning']
            }, index=[0])    
        except:
            continue

        df_generated.to_csv(f'{res_dir}/df_{index}.csv', index=False)

        index += 1
        progress_bar.update(1)
        time.sleep(entry_sleep)
    
    # Collect results
    correct = 0
    num = 0
    dataframes = []
    for i in range(questions.shape[0]):
        try:
            df_res = pd.read_csv(f'{res_dir}/df_{i}.csv')
        except:
            break
        if df_res['pred_setup'].iloc[0] == df_res['true_setup'].iloc[0]:
            correct += 1
        num += 1
        dataframes.append(df_res)

    if len(dataframes) == 0:
        return pd.DataFrame(), 0.0, 0

    df_result = pd.concat(dataframes, ignore_index=True) 
    acc = correct / num
    return df_result, acc, num


def generate_question_pairs(exp_database, n_questions=100, random_state=42):
    """
    Generate random pairs of experiments for LLM comparison.
    
    Args:
        exp_database: DataFrame with experiments
        n_questions: number of question pairs to generate
        random_state: random seed for reproducibility
    
    Returns:
        numpy array of shape (n_questions, 2) with experiment index pairs
    """
    np.random.seed(random_state)
    n_experiments = len(exp_database)
    
    questions = []
    for _ in range(n_questions):
        # Sample two different experiments
        idx_a, idx_b = np.random.choice(n_experiments, size=2, replace=False)
        questions.append([idx_a, idx_b])
    
    return np.array(questions)


def test_amide_llm_interface():
    """
    Test function to verify the amide LLM interface works.
    """
    print("Testing amide LLM interface...")
    
    # Load amide coupling data using the same function as main script
    import sys
    sys.path.append('.')
    
    # Load data using the same approach as run_amide_coupling_bo.py
    try:
        df = pd.read_csv('./amide_coupling_data/all_HTE_with_condition.csv')
        
        # Group by reactant pairs and take first group for testing
        reactant_pairs = df.groupby(['sub_1_smiles', 'sub_2_smiles'])
        first_pair_key = list(reactant_pairs.groups.keys())[0]
        first_pair_data = reactant_pairs.get_group(first_pair_key).reset_index(drop=True)
        
        print(f"Loaded {len(first_pair_data)} experiments for test reactant pair")
        amide_data = first_pair_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Generate a small number of test questions
    questions = generate_question_pairs(amide_data, n_questions=2, random_state=42)
    print(f"Generated {len(questions)} test questions")
    
    # Use first 3 experiments as context
    df_context = amide_data.head(3)
    
    try:
        # Run test survey
        df_results, accuracy, num_questions = run_amide_llm_survey_parallel(
            questions=questions,
            exp_database=amide_data,
            df_context=df_context,
            res_dir='./test_amide_llm_results',
            max_workers=1,  # Use single worker for testing
            max_questions=2  # Limit to 2 questions
        )
        
        print(f"Test completed successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Number of questions processed: {num_questions}")
        
        if len(df_results) > 0:
            print("\nFirst result:")
            print(df_results.iloc[0])
            return True
        else:
            print("No results obtained")
            return False
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_amide_llm_interface()
