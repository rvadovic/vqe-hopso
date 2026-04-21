import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from costF.costF_4q_H2_qiskit import E_exact
from costF.costF_4q_H2_qiskit import noiseless
import json
import ast
import numpy as np
from io import StringIO

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "graphs")
PRECISION = 1.59e-3
E_EXACT = E_exact.real
os.makedirs(OUTPUT_DIR, exist_ok=True)

custom_cf = ['shot_noise_5k', 'shot_noise_10k', 'noiseless']
custom_opt = ['mpi_hopso']
setting = 'cf'

cost_function_labels = {
    'noiseless': 'Noiseless',
    'shot_noise_100': '100',
    'shot_noise_1k': '1000',
    'shot_noise_5k': '5000',
    'shot_noise_10k': '10000',
    'gate_noise_1k':    '1000',
    'gate_noise_5k':    '5000',
    'gate_noise_exact': 'No shot noise',
    'gate_noise_10k': '10000'
    # ... add all your cost functions
}

# Mappings for optimizers
optimizer_labels = {
    'mpi_hopso': 'MPI HOPSO',
    'opt2': 'Optimizer 2',
    'opt3': 'Optimizer 3',
    # ... add all your optimizers
}

# Mappings for energy types
energy_type_labels = {
    'final_energy': 'Reported Energy',
    'real_energy': 'Actual Energy',
}

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

def real_energy(row):
    angles = parse_position(row['best_position'])
    if angles == None:
        return row['final_energy']
    energy_noiseless = noiseless(angles)
    return energy_noiseless

# Load all raw data into a single DataFrame
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
        print(f"Warning: could not read {fname} - {e}")
        continue

    # Ensure required column exists
    if 'final_energy' not in df.columns:
        print(f"Warning: {fname} has no 'final_energy' column – skipping")
        continue

    # Parse configuration from filename
    try:
        optimizer, costF = parse_filename(fname)
        if setting == 'cf':
            if costF not in custom_cf:
                continue
        elif setting == 'opt':
            if optimizer not in custom_opt:
                continue
    except:
        print(f"Warning: could not parse {fname} – skipping")
        continue

    # Keep only needed columns
    temp = df[['final_energy']].copy()
    temp['optimizer'] = optimizer
    temp['cost_function'] = costF
    if 'best_position' not in df.columns:
        temp['real_energy'] = temp['final_energy']
    else:
        c = []
        c.append(df[['best_position']])
        comb = pd.concat(c, ignore_index=True)
        temp['real_energy'] = comb.apply(
            lambda row: real_energy(row), axis=1
        )

    all_data.append(temp)

if not all_data:
    raise RuntimeError("No valid raw data files found.")

combined = pd.concat(all_data, ignore_index=True)
'''
combined['real_energy'] = combined.apply(
    lambda row: real_energy(row), axis=1
)
'''
optimizers = sorted(combined['optimizer'].unique())
cost_functions = sorted(combined['cost_function'].unique())

# -------------------------------
# 1. Box plots: per cost function (one figure per cost function)
# -------------------------------

def per_cf():
    for cf in cost_functions:
        subset = combined[combined['cost_function'] == cf]
        if subset.empty:
            continue
        melted = subset.melt(
            id_vars=['optimizer'],
            value_vars=['final_energy', 'real_energy'],
            var_name='energy_type',
            value_name='energy'
        )

        fig, ax1 = plt.subplots(figsize=(8, 6))

        sns.boxplot(data=melted, x='optimizer', y='energy', hue='energy_type', order=optimizers, width=0.2, showfliers=False, ax=ax1)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax1.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
        ax1.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
        ax1.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
        ax1.legend()

        #sns.boxplot(data=subset, x='optimizer', y='real_energy', order=optimizers, width=0.2, showfliers=False, ax=ax2)
        #ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        
        plt.title(f"Final Energy Distribution – Cost Function: {cf}")
        plt.ylabel("Energy (Hartree)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Sanitize filename
        safe_cf = cf.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(OUTPUT_DIR, f"boxplot_costfunc_{safe_cf}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

# -------------------------------
# 2. Box plots: per optimizer (one figure per optimizer)
# -------------------------------
def per_optimizer():
    for opt in optimizers:
        subset = combined[combined['optimizer'] == opt]
        if subset.empty:
            continue
        melted = subset.melt(
            id_vars=['cost_function'],
            value_vars=['final_energy', 'real_energy'],
            var_name='energy_type',
            value_name='energy'
        )

        fig, ax1 = plt.subplots(figsize=(8, 6))

        sns.boxplot(data=melted, x='cost_function', y='energy', hue='energy_type', order=cost_functions, width=0.2, showfliers=False, ax=ax1)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax1.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
        ax1.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
        ax1.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
        ax1.legend()

        plt.title(f"Final Energy Distribution – Optimizer: {opt}")
        plt.ylabel("Energy (Hartree)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"boxplot_optimizer_{opt}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

def custom(setting):
    if(setting == 'cf'):
        custom = custom_opt
        attr = 'optimizer'
        x = 'cost_function'
        x_label_map = cost_function_labels
        order = custom_cf
    else:
        custom = custom_cf
        attr = 'cost_function'
        x = 'optimizer'
        x_label_map = optimizer_labels
        order = custom_opt
    for i in custom:
        subset = combined[combined[attr] == i]

        melted = subset.melt(
            id_vars=[x],
            value_vars=['final_energy', 'real_energy'],
            var_name='energy_type',
            value_name='energy'
        )

        melted[x] = melted[x].map(x_label_map)
        melted['energy_type'] = melted['energy_type'].map(energy_type_labels)

        fig, ax1 = plt.subplots(figsize=(8, 6))

        labeled_order = [x_label_map[raw] for raw in order]
        
        color_dict = {'Reported Energy': 'steelblue', 'Actual Energy': 'darkorange'}
        sns.boxplot(data=melted, x=x, y='energy', hue='energy_type', order=labeled_order, width=0.2, showfliers=False, ax=ax1, palette=color_dict)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax1.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
        ax1.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
        ax1.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
        ax1.legend()

        #sns.boxplot(data=subset, x='optimizer', y='real_energy', order=optimizers, width=0.2, showfliers=False, ax=ax2)
        #ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

        plt.title(f"Shot noise (5000, 10000, Noiseless)")
        plt.ylabel("Energy (Hartree)")
        plt.xlabel("Shots")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Sanitize filename
        out_path = os.path.join(OUTPUT_DIR, f"boxplot_shot_noise_closeup.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


custom(setting)
print("All box plots generated.")
