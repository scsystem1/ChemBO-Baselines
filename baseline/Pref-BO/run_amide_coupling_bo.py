#!/usr/bin/env python3
"""
Amide Coupling Preference BO Pipeline

This script adapts the existing preference BO workflow to work with the amide coupling dataset.
It groups experiments by reactant pairs (as done in the original paper) and runs the complete
pipeline including LLM surveys and Bayesian optimization.

Key differences from original:
1. Uses amide coupling reaction data instead of cross-coupling
2. Groups experiments by substrate pairs (carboxylic acid + amine)
3. Parameters are activation reagent, additive, base, solvent instead of aryl halide, additive, base, ligand
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path

# Add local modules
sys.path.append('.')
sys.path.append('./LLM')

from model import GP_Model, train_preference, compute_probability
from acquisition import optim, acquire
from sklearn.preprocessing import StandardScaler
from gpytorch.priors import GammaPrior
import torch

# Import real LLM interface for amide coupling
from amide_llm_interface import (
    run_amide_llm_survey_parallel, 
    generate_question_pairs,
    test_amide_llm_interface
)

def load_and_prepare_amide_data():
    """
    Load amide coupling data and prepare it for BO workflow.
    
    Returns:
        dict: Dictionary with reactant pair keys and experiment data
    """
    df = pd.read_csv('./amide_coupling_data/all_HTE_with_condition.csv')
    # Debug: show top of raw dataframe and columns
    print("[DEBUG] Loaded raw data: shape=", df.shape)
    print("[DEBUG] Raw data columns:", df.columns.tolist())
    print("[DEBUG] Raw data head:\n", df.head(3).to_string(index=False))
    
    # Group by reactant pairs
    grouped_data = {}
    for (sub1, sub2), group in df.groupby(['sub_1_smiles', 'sub_2_smiles']):
        # Only use groups with sufficient data (>= 50 conditions like original paper)
        if len(group) >= 50:
            reactant_key = f"{sub1}_{sub2}"
            grouped_data[reactant_key] = {
                'reactant_pair': reactant_key,
                'sub_1_smiles': sub1,
                'sub_2_smiles': sub2,
                'experiments': group.copy(),
                'n_experiments': len(group)
            }
    
    print(f"Found {len(grouped_data)} reactant pairs with >= 50 conditions")
    
    return grouped_data

def _log_df_head(name, df, n=3):
    try:
        print(f"--- {name} (shape={getattr(df, 'shape', 'N/A')}) head:\n", df.head(n))
    except Exception as e:
        print(f"--- {name} (unable to show head): {e}")

def create_feature_encoding(experiments_df):
    """
    Create one-hot encoding for amide coupling conditions.
    
    Args:
        experiments_df: DataFrame with experiment conditions
        
    Returns:
        tuple: (X, y) where X is encoded features and y is yields
    """
    # Get categorical variables
    categorical_vars = ['Activation_ID', 'Additive_ID', 'Base_ID', 'solvent_id']
    
    # Create one-hot encoding
    encoded_dfs = []
    for var in categorical_vars:
        encoded = pd.get_dummies(experiments_df[var], prefix=var)
        encoded_dfs.append(encoded)
    
    # Combine all encodings
    X = pd.concat(encoded_dfs, axis=1)
    y = experiments_df['yield'].values
    
    print(f"Feature dimensions: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    _log_df_head('Encoded feature dataframe', pd.DataFrame(X, columns=X.columns))
    
    return X.values, y

def generate_amide_survey_questions(experiments_df, n_questions=1000, seed=42):
    """
    Generate pairwise comparison questions for amide coupling experiments.
    
    Args:
        experiments_df: DataFrame with experiments
        n_questions: Number of pairwise questions to generate
        seed: Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Array of question pairs (indices)
    """
    np.random.seed(seed)
    
    n_experiments = len(experiments_df)
    questions = []
    
    # Generate random pairs
    for _ in range(n_questions):
        idx1, idx2 = np.random.choice(n_experiments, size=2, replace=False)
        questions.append([idx1, idx2])
    
    return np.array(questions)

def create_amide_survey_prompt(idx_a, idx_b, experiments_df):
    """
    Create survey prompt for amide coupling reactions.
    
    Args:
        idx_a, idx_b: Indices of experiments to compare
        experiments_df: DataFrame with experiment data
        
    Returns:
        str: Formatted prompt for LLM
    """
    exp_a = experiments_df.iloc[idx_a]
    exp_b = experiments_df.iloc[idx_b]
    
    # Load condition mappings
    sys.path.append('./amide_coupling_data')
    from convert_conditions import conditions_map
    
    # Get condition details
    cond_a = conditions_map[exp_a['condition_id']]
    cond_b = conditions_map[exp_b['condition_id']]
    
    prompt = f"""
You are an expert organic chemist specializing in amide coupling reactions. Please predict which experimental setup will give a higher yield for the following amide coupling reaction.

General Reaction: Carboxylic Acid + Amine → Amide
Substrates: {exp_a['sub_1_smiles']} + {exp_a['sub_2_smiles']} → Amide Product

Setup A:
- Activation Reagent: {cond_a['Activation_reagent']} ({cond_a['Activation_equiv']} equiv)
- Additive: {cond_a['Additive']} ({cond_a['Additive_equiv']} equiv)
- Base: {cond_a['Base']} ({cond_a['Base_equiv']} equiv) 
- Solvent: {cond_a['Solvent']}

Setup B:
- Activation Reagent: {cond_b['Activation_reagent']} ({cond_b['Activation_equiv']} equiv)
- Additive: {cond_b['Additive']} ({cond_b['Additive_equiv']} equiv)
- Base: {cond_b['Base']} ({cond_b['Base_equiv']} equiv)
- Solvent: {cond_b['Solvent']}

Consider factors like:
1. Activation reagent efficiency for these specific substrates
2. Base strength and compatibility with activation reagent
3. Additive effects on reaction kinetics
4. Solvent effects on solubility and reaction rate
5. Substrate sterics and electronics

Output your prediction as JSON: {{"Setup": "A" or "B", "reasoning": "your detailed chemical reasoning"}}
"""
    
    return prompt

def run_llm_survey(experiments_df, questions, context_experiments=None, 
                   seed=42, res_dir='./llm_results'):
    """
    Run LLM survey using the real Claude LLM and return parsed responses.
    
    Args:
        experiments_df: DataFrame with experiments
        questions: Array of question pairs
        context_experiments: Optional context experiments passed to the LLM for context
        seed: Random seed
        res_dir: Results directory for LLM results
        
    Returns:
        numpy.ndarray: Preference comparisons for training
    """
    # Always use real LLM for preference learning. Surface errors to the caller.
    print("Using real Claude LLM for preference learning...")
    try:
        df_results, acc, num_questions = run_amide_llm_survey_parallel(
            questions=questions,
            exp_database=experiments_df,
            df_context=context_experiments,
            res_dir=res_dir,
            max_workers=3,  # Conservative for API rate limits
            entry_sleep=2   # Sleep between requests
        )
    except Exception as e:
        raise RuntimeError(f"LLM survey failed: {e}") from e

    # If LLM returned no rows, raise an error (no silent fallback)
    if df_results is None or len(df_results) == 0:
        raise RuntimeError("LLM survey returned no responses (df_results is empty).")

    print(f"LLM Survey completed: {acc:.3f} accuracy on {num_questions} questions")

    # Convert LLM results to comparison format
    comparisons = []
    for _, row in df_results.iterrows():
        q_idx = row['question']
        pred_setup = row['pred_setup']

        idx_a, idx_b = questions[q_idx]

        # Store as (better_idx, worse_idx) based on LLM prediction
        if pred_setup == 'A':
            comparisons.append([idx_a, idx_b])
        else:
            comparisons.append([idx_b, idx_a])

    return np.array(comparisons)

def run_preference_bo_experiment(reactant_data, n_trials=5, n_start=5, n_iterations=50, 
                                method='pibo', track_utility=False):
    """
    Run preference-guided Bayesian optimization for a single reactant pair.
    
    Args:
        reactant_data: Dictionary with experiment data for one reactant pair
        n_trials: Number of BO trials to run
        n_start: Number of initial experiments
        n_iterations: Number of BO iterations
        method: Acquisition function ('pibo', 'ei', 'ucb', 'random')
        track_utility: If True, return utility data for Figure 2 (only for pibo)
        
    Returns:
        numpy.ndarray: Results array with shape (n_trials, n_iterations+1)
        dict (optional): Utility data if track_utility=True and method='pibo'
    """
    experiments_df = reactant_data['experiments'].reset_index(drop=True)
    
    # Create feature encoding
    X, y = create_feature_encoding(experiments_df)
    
    # Only run LLM survey for preference-based methods (pibo)
    if method == 'pibo':
        # Generate survey questions and responses
        questions = generate_question_pairs(experiments_df, n_questions=100)
        context_experiments = experiments_df.head(5)
        print(f"Generated {len(questions)} question pairs. Example questions (first 5): {questions[:5]}")
        _log_df_head('Context experiments', context_experiments)

        # Run LLM survey (always real LLM)
        comparisons = run_llm_survey(
            experiments_df=experiments_df,
            questions=questions,
            context_experiments=context_experiments,
            res_dir=f'./llm_results_{reactant_data["reactant_pair"]}'
        )
        print(f"LLM returned {len(comparisons)} comparisons. Example (first 5): {comparisons[:5]}")
        
        # Train preference model
        X_pref = X[questions[:100].flatten()]  # Use subset for preference training
        pref_model = train_preference(x_train=X_pref, train_comp=comparisons[:100])
        
        # Compute preference probabilities for all experiments
        all_pi = compute_probability(model=pref_model, all_x=X)
        print(f"Computed preference probabilities for {len(all_pi)} experiments. Example (first 5): {all_pi[:5]}")
    else:
        # For EI, UCB, random: no LLM needed
        print(f"[INFO] Using {method.upper()} - no LLM survey needed")
        all_pi = None  # Will not be used
    
    # Run BO trials
    all_results = []
    
    from tqdm import tqdm
    print(f"\n[INFO] Running {n_trials} BO trials with {n_iterations} iterations each...")
    
    for trial in tqdm(range(n_trials), desc="BO Trials", leave=True):
        np.random.seed(trial)
        
        # Initial experiment selection: preference-guided for PIBO, random for others
        n_experiments = len(experiments_df)
        all_idx = np.arange(n_experiments)
        
        if method == 'pibo':
            # Use LLM preferences to select high-quality initial experiments
            top_pref_idx = np.argsort(all_pi)[-n_start*3:]  # Top candidates
            done_idx = np.random.choice(top_pref_idx, size=n_start, replace=False)
        else:
            # Random initial selection for EI, UCB, random
            done_idx = np.random.choice(all_idx, size=n_start, replace=False)
        
        remaining_idx = np.array([i for i in all_idx if i not in done_idx])
        
        # Track best yield over time
        results = np.zeros(n_iterations + 1)
        results[0] = np.max(y[done_idx])
        
        # BO iterations
        for t in range(n_iterations):
            # Prepare training data
            scaler = StandardScaler()
            y_train = y[done_idx].reshape(-1, 1)
            if t > 0:
                y_train = scaler.fit_transform(y_train).flatten()
            else:
                y_train = y_train.flatten()  # Ensure it's 1D
            
            x_train = torch.tensor(X[done_idx], dtype=torch.float64)
            y_train_torch = torch.tensor(y_train, dtype=torch.float64)
            
            # Train surrogate model (using same hyperparameters as original)
            surrogate = GP_Model(
                x_train, y_train_torch, gpu=False, nu=2.5,
                noise_constraint=1e-5,
                lengthscale_prior=[GammaPrior(2.0, 0.2), 5.0],
                outputscale_prior=[GammaPrior(5.0, 0.5), 8.0],
                noise_prior=[GammaPrior(1.5, 0.5), 1.0],
                n_restarts=0, learning_rate=0.1, training_iters=100
            )
            surrogate.fit()
            
            # Acquisition function
            opt = optim(
                surrogate=surrogate,
                input_space=torch.tensor(X[remaining_idx], dtype=torch.float64),
                method=method,
                preference=all_pi[remaining_idx] if method == 'pibo' else None
            )
            
            # Create args for acquisition
            class Args:
                def __init__(self):
                    self.fmax = np.max(y_train) if t > 0 else np.max(y[done_idx])
                    self.beta = 10.0  # Increased for stronger preference signal
                    self.iter = t + 1
            
            args = Args()
            
            # Select next experiment
            temp_idx = acquire(opt, args)
            new_idx = remaining_idx[temp_idx]
            
            # Update indices
            remaining_idx = np.delete(remaining_idx, temp_idx)
            done_idx = np.append(done_idx, new_idx)
            
            # Record best yield
            results[t + 1] = np.max(y[done_idx])
        
        all_results.append(results)
    
    # Log summary statistics
    results_array = np.array(all_results)
    final_yields = results_array[:, -1]
    initial_yields = results_array[:, 0]
    print(f"\n[INFO] Completed {n_trials} trials")
    print(f"  Initial yield: {initial_yields.mean():.3f} ± {initial_yields.std():.3f}")
    print(f"  Final yield:   {final_yields.mean():.3f} ± {final_yields.std():.3f}")
    print(f"  Improvement:   {(final_yields.mean() - initial_yields.mean()):.3f}")
    
    # Compute utility data if requested (only for PIBO with LLM)
    if track_utility and method == 'pibo' and all_pi is not None:
        # Extract latent utility function values from preference GP model
        # According to Chu & Ghahramani 2005, these are the f(x) values
        # that define the preference relations via P(x1 > x2) = Phi((f(x1)-f(x2))/sqrt(2*sigma))
        x_all = torch.tensor(X, dtype=torch.float64)
        posterior = pref_model.posterior(x_all)
        all_utility = posterior.mean.detach().numpy().flatten()
        
        # Compute Pearson correlation
        from scipy import stats
        pearson_r, pearson_p = stats.pearsonr(all_utility, y)
        
        utility_data = {
            'true_yields': y.copy(),
            'utility_values': all_utility.copy(),
            'preference_probabilities': all_pi.copy(),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'reactant_pair': reactant_data['reactant_pair'],
            'n_experiments': len(y)
        }
        
        print(f"\n[INFO] Utility Tracking:")
        print(f"  Pearson r: {pearson_r:.4f}")
        print(f"  P-value: {pearson_p:.2e}")
        
        return results_array, utility_data
    
    return results_array

def main():
    parser = argparse.ArgumentParser(description='Run Preference BO on Amide Coupling Data')
    parser.add_argument('--reactant_idx', type=int, default=0, 
                       help='Index of reactant pair to test (0-based)')
    parser.add_argument('--method', type=str, default='pibo',
                       choices=['pibo', 'ei', 'ucb', 'random'],
                       help='Acquisition method')
    parser.add_argument('--n_trials', type=int, default=5,
                       help='Number of BO trials')
    parser.add_argument('--n_iterations', type=int, default=50,
                       help='Number of BO iterations per trial')
    # Note: this pipeline now always uses the real Claude LLM
    parser.add_argument('--test_llm', action='store_true',
                       help='Test LLM interface before running full experiment')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading amide coupling data...")
    grouped_data = load_and_prepare_amide_data()
    
    # Test LLM interface if requested
    if args.test_llm:
        print("Testing LLM interface...")
        if test_amide_llm_interface():
            print("LLM interface test passed!")
            print("Proceeding with full experiment...")
        else:
            print("LLM interface test failed!")
            return
    
    # Get reactant pairs with sufficient data
    reactant_keys = list(grouped_data.keys())
    
    if args.reactant_idx >= len(reactant_keys):
        print(f"Error: reactant_idx {args.reactant_idx} >= {len(reactant_keys)}")
        return
    
    reactant_key = reactant_keys[args.reactant_idx]
    reactant_data = grouped_data[reactant_key]
    
    print(f"\nRunning BO for reactant pair {args.reactant_idx}: {reactant_key}")
    print(f"Number of experiments: {reactant_data['n_experiments']}")
    print(f"Method: {args.method}")
    print(f"Trials: {args.n_trials}")
    # Run experiment (always using real LLM)
    results = run_preference_bo_experiment(
        reactant_data, 
        n_trials=args.n_trials,
        n_iterations=args.n_iterations,
        method=args.method
    )
    
    # Save results in secure NumPy format (no pickle)
    output_file = f"{args.output_dir}/amide_bo_results_{reactant_key}_{args.method}.npz"
    
    # Save numerical results as numpy
    np.savez(
        output_file,
        results=results,
        n_experiments=reactant_data['n_experiments']
    )
    
    # Save metadata as JSON (secure, human-readable)
    metadata_file = f"{args.output_dir}/amide_bo_metadata_{reactant_key}_{args.method}.json"
    metadata = {
        'reactant_key': reactant_key,
        'sub_1_smiles': reactant_data['sub_1_smiles'],
        'sub_2_smiles': reactant_data['sub_2_smiles'],
        'n_experiments': int(reactant_data['n_experiments']),
        'method': args.method,
        'n_trials': args.n_trials,
        'n_iterations': args.n_iterations
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Final best yields (mean ± std): {results[:, -1].mean():.3f} ± {results[:, -1].std():.3f}")
    
    # Save summary statistics
    summary_file = f"{args.output_dir}/summary_{reactant_key}_{args.method}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Amide Coupling Preference BO Results\n")
        f.write(f"=====================================\n\n")
        f.write(f"Reactant Pair: {reactant_key}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Trials: {args.n_trials}\n")
        f.write(f"Iterations: {args.n_iterations}\n")
        f.write(f"LLM Type: Real Claude\n")
        f.write(f"LLM Accuracy: N/A (real LLM used)\n\n")
        f.write(f"Results Summary:\n")
        f.write(f"Initial best yield (mean ± std): {results[:, 0].mean():.3f} ± {results[:, 0].std():.3f}\n")
        f.write(f"Final best yield (mean ± std): {results[:, -1].mean():.3f} ± {results[:, -1].std():.3f}\n")
        f.write(f"Improvement: {results[:, -1].mean() - results[:, 0].mean():.3f}\n")

if __name__ == "__main__":
    main()