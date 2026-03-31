import os
import csv
import pandas as pd
import numpy as np
from src.costF.costF_4q_H2_qiskit import E_exact

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "summaries")
PRECISION = 1.59e-3
E_EXACT = E_exact.real

def write_to_csv(costF_name, optimzer_name, results):
    out_path = os.path.join(DATA_DIR, f'{optimzer_name}_{costF_name}.csv')
    f = open(out_path, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

def parse_filename(filename):
    base = filename.replace('.csv', '')
    parts = base.split('_')
    i = 1
    part0 = parts[0]

    if(part0 == "mpi"):
        optimizer = '_'.join(parts[:2])
        i+=1

    costF = '_'.join(parts[i:])
    return optimizer, costF

def rename_cols(agg_df, prefix=''):
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.rename(columns={
        'final_energy_median': 'energy_median',
        'final_energy_min': 'energy_min',
        'final_energy_max': 'energy_max',
        'final_energy_<lambda_0>': 'energy_q25',
        'final_energy_<lambda_1>': 'energy_q75',
        'error_median': 'error_median',
        'error_min': 'error_min',
        'error_max': 'error_max',
        'error_<lambda_0>': 'error_q25',
        'error_<lambda_1>': 'error_q75',
        'success_mean': 'success_rate',
    })
    if prefix:
        agg_df = agg_df.add_prefix(prefix)
    return agg_df

def analyze():
    all_data = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.csv'):
            continue
        filepath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(filepath)
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

    agg_funcs = {
        'final_energy': ['median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'error': ['median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'success': 'mean',
        'time': 'median'
    }

    cost_funcs = combined['costF'].unique()
    for cf in cost_funcs:
        subset = combined[combined['costF'] == cf]

        grouped = subset.groupby('optimizer').agg(agg_funcs)
        grouped = rename_cols(grouped)
        grouped = grouped.reset_index()

        out_path = os.path.join(OUTPUT_DIR, f'summary_{cf}.csv')
        grouped.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

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