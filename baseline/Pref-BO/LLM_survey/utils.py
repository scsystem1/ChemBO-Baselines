import json
from tqdm import tqdm
import os
import sys
import concurrent.futures
import pandas as pd
import numpy as np
import warnings
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from kimi_client import call_kimi_chat, parse_jsonish_response

warnings.filterwarnings('ignore')


def call_bedrock_claude(user_prompt = None, model_type =  0 ):
    del model_type
    return call_kimi_chat(user_prompt)


def get_prompt(q_idx, questions , exp_database , df_context, task_reaction ):
    
      
    idx_a = questions[q_idx][0]
    idx_b = questions[q_idx][1]
    
    
    
    if df_context is None:
        query = "You can predict reaction yield."        
    else:
        query = "You can predict reaction yield using the following examples and your chemistry knoweldge.\n"
        query += "Examples:\n" 
        query += "Entry, Aryl_halide, Additive, Base, Ligand, Yield\n"
        for i in range(df_context.shape[0]):
            query += f"""{df_context['Aryl_halide_SMILES'][i]}, {df_context['Additive_SMILES'][i]}, {df_context['Base_SMILES'][i]}, {df_context['Ligand_SMILES'][i]}, {df_context['yield'][i]}\n"""
        
    
    
    query += f"""\nTask: For the following reaction predict which experiment setup leads to a higher yield and output the response (A or B) and the reasoning in JSON format.
    Task reaction: {task_reaction}
    For any reaction an additive, a base, and a palladium-ligand are present for any given reaction

    Setup A:
    Aryl halide: {exp_database['Aryl_halide_SMILES'][idx_a]}
    Additive: {exp_database['Additive_SMILES'][idx_a]}
    Base: {exp_database['Base_SMILES'][idx_a]}
    Ligand: {exp_database['Ligand_SMILES'][idx_a]}
    
    
    Setup B:
    Aryl halide: {exp_database['Aryl_halide_SMILES'][idx_b]}
    Additive: {exp_database['Additive_SMILES'][idx_b]}
    Base: {exp_database['Base_SMILES'][idx_b]}
    Ligand: {exp_database['Ligand_SMILES'][idx_b]}
    
    Output: Reaction setup (A or B) with higher yield and the reasoning in JSON object with 'Setup' and 'reasoning' keys
    """
    
    return query

def get_prompt_DA(q_idx, questions , exp_database , df_context, task_reaction):
    
    idx_a = questions[q_idx][0]
    idx_b = questions[q_idx][1]
    
    
    if df_context is None:
        query = "You can predict reaction yield."        
    else:
        query = "You can predict reaction yield using the following examples and your chemistry knoweldge.\n"
        query += "Examples:\n" 
        query += "Entry, Base, Ligand, Solvent, Concentration, Temperature, Yield\n"
        for i in range(df_context.shape[0]):
            query += f"""{df_context['Base_SMILES'][i]}, {df_context['Ligand_SMILES'][i]}, {df_context['Solvent_SMILES'][i]}, {df_context['Concentration'][i]},{df_context['Temp_C'][i]},{df_context['yield'][i]}\n"""
        
    
    
    query += f"""\nTask: For the following reaction predict which experiment setup leads to a higher yield and output the response (A or B) and the reasoning in JSON format.
    Task reaction: {task_reaction}
    For any reaction a base, a ligand, a solvent, concentration, and temperature are present for any given reaction

    Setup A:
    Base: {exp_database['Base_SMILES'][idx_a]}
    Ligand: {exp_database['Ligand_SMILES'][idx_a]}
    Solvent: {exp_database['Solvent_SMILES'][idx_a]}
    Concentration: {exp_database['Concentration'][idx_a]}
    Temperature: {exp_database['Temp_C'][idx_a]}
    
    
    Setup B:
    Base: {exp_database['Base_SMILES'][idx_b]}
    Ligand: {exp_database['Ligand_SMILES'][idx_b]}
    Solvent: {exp_database['Solvent_SMILES'][idx_b]}
    Concentration: {exp_database['Concentration'][idx_b]}
    Temperature: {exp_database['Temp_C'][idx_b]}
    
    Output: Reaction setup (A or B) with higher yield and the reasoning in JSON object with 'Setup' and 'reasoning' keys
    """
    
    return query


def process_single_question(args):
    index, questions, exp_database, df_context, task_reaction, res_dir, entry_sleep, time_out_sleep = args
    
    max_retries = 5
    for attempt in range(max_retries):
        prompt_molecule = get_prompt(index, questions, exp_database, df_context, task_reaction)
        
        try:
            response_first = call_bedrock_claude(prompt_molecule)
        except:
            time.sleep(60)
            continue

        yield_a = exp_database['yield'][questions[index][0]]
        yield_b = exp_database['yield'][questions[index][1]]
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
            df_generated = pd.DataFrame({'question': index, 'pred_setup': generated_data['Setup'], 'true_setup': true_setup, 'reasoning': generated_data['reasoning']}, index=[0])
        except:
            continue

        df_generated.to_csv(f'{res_dir}/df_{index}.csv', index=False)
        time.sleep(entry_sleep)
        
        return df_generated

    # If all retries fail, return None
    return None


def run_llm_survey_parallel(index=0, questions=None, exp_database=None, df_context=None, task_reaction=None, res_dir='./results_survey/r1_aa1', entry_sleep=1, time_out_sleep=60, max_workers=5):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    args_list = [(i, questions, exp_database, df_context, task_reaction, res_dir, entry_sleep, time_out_sleep) 
                 for i in range(index, questions.shape[0])]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_single_question, args): args[0] for args in args_list}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(args_list), desc="Processing"):
            index = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f'Question {index} generated an exception: {exc}')

    df_result = pd.concat(results, ignore_index=True)
    correct = sum(df_result['pred_setup'] == df_result['true_setup'])
    num = len(df_result)
    acc = correct / num if num > 0 else 0

    return df_result, acc, num











def run_llm_survey(index = 0, questions = None, exp_database = None, df_context = None, task_reaction = None,  res_dir =  './results_survey/r1_aa1', entry_sleep = 1, time_out_sleep = 60):
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    progress_bar = tqdm(total= questions.shape[0], desc="Processing")
    
    while index < questions.shape[0]:
    
        prompt_molecule = get_prompt(index, questions , exp_database , df_context, task_reaction)
        # prompt_molecule = get_prompt_DA(index, questions , exp_database , df_context, task_reaction)
        # print(prompt_molecule)
        try:
            response_first = call_bedrock_claude(prompt_molecule)
        except:
            time.sleep(60)


        yield_a = exp_database['yield'][questions[index][0]]
        yield_b = exp_database['yield'][questions[index][1]]

        if yield_a >= yield_b:
            true_setup = 'A'
        else:
            true_setup = 'B'

        query_prompt = f"""
        From the following return only the predicted yield in Json format with 'Setup' and 'reasoning' columns.

        Input: {response_first}"""
        query_prompt +="""
        Output template:
        {
      "Setup":  ,
      "reasoning": 
        }
        """

        try:
            response = call_bedrock_claude(query_prompt, model_type= 2)

        except:
            time.sleep(time_out_sleep)

        if "\n\n" in response:
            ans = response.split("\n\n")[1]
        else:

            try:
                ans = response.split("json\n")[1]
            except:
                ans = response

        # ans = ans.replace("\\", r"\\")
        try:
            generated_data = parse_jsonish_response(ans)
        except:
            continue

        try:
            dummy = generated_data
        except:
            # print(f"index {index} failed format")
            continue


        try:
            df_generated = pd.DataFrame({'question': index, 'pred_setup': generated_data['Setup'], 'true_setup': true_setup, 'reasoning': generated_data['reasoning'] }, index = [0])    

        except:
            # print(f"index {index} df generation failed")
            # print(generated_data)
            continue

        df_generated.to_csv(f'{res_dir}/df_{index}.csv', index = False)


        # print(f"index {index} Done", generated_data['Setup'], true_setup, generated_data['reasoning'][:50])
        index += 1
        progress_bar.update(1)
        time.sleep(entry_sleep)
    
    correct = 0
    num = 0
    dataframes = []
    for i in range(questions.shape[0]):
        try:
            df_res = pd.read_csv(f'{res_dir}/df_{i}.csv')
        except:
            break
        if df_res['pred_setup'][0] == df_res['true_setup'][0]:
            correct += 1
        num += 1
        dataframes.append(df_res)


    df_result = pd.concat(dataframes, ignore_index=True) 
    acc = correct/num
    return df_result, acc, num
