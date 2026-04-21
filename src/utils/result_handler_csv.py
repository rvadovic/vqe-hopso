import os
import csv
import pandas as pd
import numpy as np
import ast
from src.costF.costF_4q_H2_qiskit import E_exact
from src.costF.costF_4q_H2_qiskit import noiseless
from io import StringIO
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "summaries")
PRECISION = 1.59e-3
E_EXACT = E_exact.real

def write_energies_csv(name, energies):
    out_path = os.path.join(DATA_DIR, f'{name}.csv')
    with open(out_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(energies)

def write_to_csv(costF_name, optimzer_name, results):
    out_path = os.path.join(DATA_DIR, f'{optimzer_name}_{costF_name}.csv')
    serialised = []
    for row in results:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (list, np.ndarray)):
                clean[k] = json.dumps(v if isinstance(v, list) else v.tolist())
            else:
                clean[k] = v
        serialised.append(clean)
 
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=serialised[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(serialised)

def parse_filename(filename):
    base = filename.replace('.csv', '')
    parts = base.split('_')
    i = 1
    part0 = parts[0]
    optimizer = part0
    if(part0 == "mpi"):
        optimizer = '_'.join(parts[:2])
        i+=1

    costF = '_'.join(parts[i:])
    return optimizer, costF

def rename_cols(agg_df, prefix=''):
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.rename(columns={
        'final_energy_mean': 'energy_mean',
        'final_energy_median': 'energy_median',
        'final_energy_min': 'energy_min',
        'final_energy_max': 'energy_max',
        'final_energy_<lambda_0>': 'energy_q25',
        'final_energy_<lambda_1>': 'energy_q75',
        'error_mean': 'error_mean',
        'error_median': 'error_median',
        'error_min': 'error_min',
        'error_max': 'error_max',
        'error_<lambda_0>': 'error_q25',
        'error_<lambda_1>': 'error_q75',
        'success_mean': 'success_rate',
        'energy_diff_median': 'energy_diff_median',
        'energy_diff_min': 'energy_diff_min',
        'energy_diff_max': 'energy_diff_max',
        'energy_diff_<lambda_0>': 'energy_diff_q25',
        'energy_diff_<lambda_1>': 'energy_diff_q75'
    })
    if prefix:
        agg_df = agg_df.add_prefix(prefix)
    return agg_df
def parse_position(val):
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    # Try JSON first (new format), fall back to ast.literal_eval (old format)
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(val)

def energy_diff(row):
    angles = parse_position(row['best_position'])
    if angles == None:
        return 0.0
    energy_noiseless = noiseless(angles)
    return abs(abs(energy_noiseless) - abs(row['final_energy']))

def better_than_found(row):
    angles = parse_position(row['best_position'])
    if angles == None:
        return False
    energy_noiseless = noiseless(angles)
    return abs(abs(energy_noiseless) - abs(E_EXACT)) < abs(abs(row['final_energy']) - abs(E_EXACT))

def actual_success(row):
    angles = parse_position(row['best_position'])
    if angles == None:
        return False
    energy_noiseless = noiseless(angles)
    return abs(abs(energy_noiseless) - abs(E_EXACT)) <= PRECISION

def analyze():
    all_data = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.csv'):
            continue
        filepath = os.path.join(DATA_DIR, fname)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            df = pd.read_csv(StringIO(''.join(lines)), sep=';')
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        # Parse configuration from filename
        try:
            optimizer, costF = parse_filename(fname)
        except ValueError as e:
            print(f"Skipping {fname}: {e}")
            continue

        df['optimizer'] = optimizer
        df['costF'] = costF

        all_data.append(df)

    if not all_data:
        print("No valid data files found.")
        exit(1)

    combined = pd.concat(all_data, ignore_index=True)

    combined['error'] = abs(abs(combined['final_energy']) - abs(E_EXACT))
    combined['success'] = combined['error'] <= PRECISION

    combined['energy_diff'] = combined.apply( 
        lambda row: energy_diff(row), axis=1
    )

    combined['better_than_found'] = combined.apply(
        lambda row: better_than_found(row), axis=1
    )

    combined['actual_success'] = combined.apply(
        lambda row: actual_success(row), axis=1
    )

    agg_funcs = {
        'final_energy': ['mean', 'median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'error': ['mean', 'median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'success': 'mean',
        'time': 'median',
        'energy_diff': ['mean', 'median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'better_than_found': 'mean',
        'actual_success': 'mean'
    }
    """
    cost_funcs = combined['costF'].unique()
    for cf in cost_funcs:
        subset = combined[combined['costF'] == cf]

        grouped = subset.groupby('optimizer').agg(agg_funcs)
        grouped = rename_cols(grouped)
        grouped = grouped.reset_index()

        out_path = os.path.join(OUTPUT_DIR, f'summary_{cf}.csv')
        grouped.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
    """
    optimizers = combined['optimizer'].unique()
    for opt in optimizers:
        subset = combined[combined['optimizer'] == opt]
        
        grouped = subset.groupby('costF').agg(agg_funcs)
        grouped = rename_cols(grouped)
        grouped = grouped.reset_index()

        out_path = os.path.join(OUTPUT_DIR, f'summary_{opt}.csv')
        grouped.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
    print("Done.")