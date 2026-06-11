import os
from datetime import datetime

import numpy as np
import pandas as pd


def save_individual_trial(trial_results, objective_name, n_initial_points, base_dir=None):
    if base_dir is None:
        base_dir = f'results_{datetime.now().strftime("%Y%m%d")}_{n_initial_points}_init'
    os.makedirs(base_dir, exist_ok=True)

    # Define result file name
    if 'method' in trial_results[0] and trial_results[0]['method'] == 'recommended':
        filename = f'{objective_name}_recommended_results.xlsx'
    else:
        kernel = trial_results[0]['kernel']
        acq = trial_results[0]['acquisition']
        filename = f'{objective_name}_{kernel.value}_{acq.value}_results.xlsx'

    all_values = np.array([r['best_values'] for r in trial_results])
    excel_path = os.path.join(base_dir, filename)
    with pd.ExcelWriter(excel_path) as writer:
        # Statistics sheet
        stats_data = [
            ['iteration'] + list(range(len(all_values[0]))),
            ['mean'] + np.mean(all_values, axis=0).tolist(),
            ['std'] + np.std(all_values, axis=0).tolist()
        ]
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='statistics', index=False, header=False)

        # Combined seeds sheet
        combined_data = [['iteration'] + list(range(len(all_values[0])))]
        for i, result in enumerate(trial_results):
            combined_data.append([f'seed_{i}'] + result['best_values'])

        pd.DataFrame(combined_data).to_excel(writer, sheet_name='combined_seeds', index=False, header=False)

        # Individual seed sheets
        for i, result in enumerate(trial_results):
            pd.DataFrame([
                ['iteration'] + result['iterations'],
                [f'seed_{i}'] + result['best_values']
            ]).to_excel(writer, sheet_name=f'seed_{i}', index=False, header=False)


def save_recommendation_log(objective_name, seed, kernel, acquisition, n_init_sample, iteration, base_dir=None):
    if base_dir is None:
        base_dir = f'results_{datetime.now().strftime("%Y%m%d")}'
    os.makedirs(base_dir, exist_ok=True)

    log_path = os.path.join(base_dir, f'{objective_name}_recommendation_log_{seed}.xlsx')

    log_data = pd.DataFrame([
        ['iteration', 'recommended_kernel', 'recommended_acquisition'],
        [iteration, kernel, acquisition]
    ])

    # Add if already exists file log
    if os.path.exists(log_path) and iteration != n_init_sample:
        existing_log = pd.read_excel(log_path, header=None)
        log_data = pd.concat([existing_log, log_data.iloc[1:]], ignore_index=True)

    log_data.to_excel(log_path, index=False, header=False)


# save final data
def save_final_data_to_excel(train_x, train_y, seed, kernel_type, acquisition_type, objective, base_dir):

    if base_dir is None:
        base_dir = f'results_{datetime.now().strftime("%Y%m%d")}'

    os.makedirs(base_dir, exist_ok=True)
    data_dict = {}
    for i in range(train_x.shape[1]):
        data_dict[f'x{i + 1}'] = train_x[:, i].cpu().numpy()
    data_dict['function_value'] = train_y.cpu().numpy()

    if isinstance(objective, str):
        # for HPOB dataset
        objective_name = objective
    else:
        # for synthetic benchmark functions
        objective_name = objective.__name__

    # distance form original, just for reference, only for synthetic benchmark functions, otherwise distance from origin
    if objective_name == 'Levy' or objective_name == 'Rosenbrock':
        global_optimum = np.ones(train_x.shape[1])
    else:
        global_optimum = np.zeros(train_x.shape[1])
    distances = np.sqrt(np.sum((train_x.cpu().numpy() - global_optimum) ** 2, axis=1))
    data_dict['distance_to_optimum'] = distances
    best_idx = train_y.argmin().item()
    data_dict['is_best_point'] = [1 if i == best_idx else 0 for i in range(len(train_y))]

    # save result
    df = pd.DataFrame(data_dict)
    metadata = pd.DataFrame({
        'Parameter': ['Objective', 'Kernel', 'Acquisition', 'Seed', 'Best Value', 'Distance to Optimum'],
        'Value': [
            objective_name,
            kernel_type.value,
            acquisition_type.value,
            seed,
            train_y.min().item(),
            distances[best_idx]
        ]
    })

    excel_path = os.path.join(
        base_dir,
        f"{objective_name}_seed{seed}_{kernel_type.value}_{acquisition_type.value}_final.xlsx"
    )
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        df.to_excel(writer, sheet_name='Points', index=False)

    return excel_path

    # 엑셀 파일로 저장
    df.to_excel(file_path, index=False)
    print(f"Final data saved to {file_path}")

def save_acquisition_log_to_excel(log_data, acquisition_names, seed, objective, base_dir):
    """
    Saves the acquisition function selection log to an Excel file.
    """
    if not log_data:
        print("Acquisition log is empty. Nothing to save.")
        return

    file_name = f"{objective}_acquisition_log_seed_{seed}.xlsx"
    file_path = os.path.join(base_dir, file_name)

    columns = ['iteration', 'chosen_acquisition'] + [f'prob_{name}' for name in acquisition_names]
    df = pd.DataFrame(log_data, columns=columns)

    # 엑셀 파일로 저장
    df.to_excel(file_path, index=False)
    print(f"Acquisition log saved to {file_path}")


def save_kernel_log_to_excel(log_data, seed, objective, base_dir):
    """
    Saves the acquisition function selection log to an Excel file.
    """
    if not log_data:
        print("Kernel log is empty. Nothing to save.")

    file_name = f"{objective}_kernel_log_seed_{seed}.xlsx"
    file_path = os.path.join(base_dir, file_name)

    columns = ['iteration', 'chosen_kernel']
    df = pd.DataFrame(log_data, columns=columns)

    df.to_excel(file_path, index=False)
    print(f"Acquisition log saved to {file_path}")

