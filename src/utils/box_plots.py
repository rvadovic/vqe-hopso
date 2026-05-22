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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "graphs_temp")
PRECISION = 1.59e-3
E_EXACT = E_exact.real
os.makedirs(OUTPUT_DIR, exist_ok=True)

custom_cf_mitigate = ['gate_noise_5k', 'gate_noise_zne_linear_5k_4p', 'gate_noise_zne_richardson_5k', 'gate_noise_zne_exponential_5k','gate_noise_exact', 'gate_noise_zne_linear_exact_4p', 'gate_noise_zne_richardson_exact', 'gate_noise_zne_exponential_exact']
custom_cf_mitiq = [
    'gate_noise_zne_richardson_5k',
    'gate_noise_zne_richardson_exact',
    'gate_noise_zne_linear_5k',
    'gate_noise_zne_linear_exact',
    'gate_noise_zne_mitiq_linear_5k', 
    'gate_noise_zne_mitiq_richardson_5k', 
    'gate_noise_zne_mitiq_linear_exact', 
    'gate_noise_zne_mitiq_richardson_exact'
            ]
custom_cf_12particles = [
    'gate_noise_zne_richardson_5k',
    'gate_noise_zne_richardson_exact',
    'gate_noise_zne_exponential_5k',
    'gate_noise_zne_exponential_exact', 
    'gate_noise_zne_linear_5k',
    'gate_noise_zne_linear_exact',
    'gate_noise_zne_linear_5k_12particles',
    'gate_noise_zne_linear_exact_12particles',
    'gate_noise_zne_richardson_5k_12particles',
    'gate_noise_zne_richardson_exact_12particles',
    'gate_noise_zne_exponential_5k_12particles',
    'gate_noise_zne_exponential_exact_12particles'
]
custom_cf_backends = [
    'fakeAthensV2_exact',
    'fakeBogotaV2_exact',
    'fakeManilaV2_exact',
    'fakeAthensV2_5k',
    'fakeBogotaV2_5k',
    'fakeManilaV2_5k',
    ]
custom_cf = [
    'gate_noise_real_5k',
    'gate_noise_zne_linear_5k_real',
    'gate_noise_zne_richardson_5k_real',
    'gate_noise_zne_exponential_5k_real',
    'gate_noise_real_exact',
    'gate_noise_zne_linear_exact_real',
    'gate_noise_zne_richardson_exact_real',
    'gate_noise_zne_exponential_exact_real',
]
custom_cf_all = [
    'gate_noise_5k',
    'gate_noise_zne_linear_5k',
    'gate_noise_zne_richardson_5k',
    'gate_noise_zne_exponential_5k',
    'gate_noise_exact',
    'gate_noise_zne_linear_exact',
    'gate_noise_zne_richardson_exact',
    'gate_noise_zne_exponential_exact',
    'gate_noise_default_5k',
    'gate_noise_zne_linear_5k_def',
    'gate_noise_zne_richardson_5k_def',
    'gate_noise_zne_exponential_5k_def',
    'gate_noise_default_exact',
    'gate_noise_zne_linear_exact_def',
    'gate_noise_zne_richardson_exact_def',
    'gate_noise_zne_exponential_exact_def',
    'gate_noise_real_5k',
    'gate_noise_zne_linear_5k_real',
    'gate_noise_zne_richardson_5k_real',
    'gate_noise_zne_exponential_5k_real',
    'gate_noise_real_exact',
    'gate_noise_zne_linear_exact_real'
    'gate_noise_zne_richardson_exact_real',
    'gate_noise_zne_exponential_exact_real',
]

custom_opt = ['mpi_hopso']
setting = 'cf' # cf or opt

cost_function_labels = {
    'noiseless': 'Noiseless',
    'shot_noise_100': '100',
    'shot_noise_1k': '1000',
    'shot_noise_5k': 'Shot noise 5000',
    'shot_noise_10k': '10000',
    'gate_noise_1k':    'Gate noise 1000',
    'gate_noise_5k':    'D1, 5k shots',
    'gate_noise_exact': 'D1, no shots',
    'gate_noise_10k': 'Gate noise 10000',
    'gate_noise_zne_linear_5k': 'D1, Linear, 5k shots',
    'gate_noise_zne_richardson_5k': 'D1, Richardson, 5000',
    'gate_noise_zne_exponential_5k': 'D1, Exponential, 5000',
    'gate_noise_zne_linear_exact': 'D1, Linear, no shots',
    'gate_noise_zne_richardson_exact': 'D1, Richardson, no shots',
    'gate_noise_zne_exponential_exact': 'D1, Exponential, no shots',
    'gate_noise_zne_mitiq_linear_5k': 'Mitiq, 5000',
    'gate_noise_zne_mitiq_richardson_5k': 'Mitiq, 5000',
    'gate_noise_zne_mitiq_exponential_5k': 'Mitiq, 5000',
    'gate_noise_zne_mitiq_linear_exact': 'Mitiq, no shots',
    'gate_noise_zne_mitiq_richardson_exact': 'Mitiq, no shots',
    'gate_noise_zne_mitiq_exponential_exact': 'Mitiq, no shots',
    'gate_noise_zne_linear_5k_12particles': '5000, 12 particles',
    'gate_noise_zne_linear_exact_12particles': 'no shots, 12 particles',
    'gate_noise_zne_richardson_5k_12particles': '5000, 12 particles',
    'gate_noise_zne_richardson_exact_12particles': 'no shots, 12 particles',
    'gate_noise_zne_exponential_5k_12particles': '5000, 12 particles',
    'gate_noise_zne_exponential_exact_12particles': 'no shots, 12 particles',
    'gate_noise_5k_postzne_exponential_5k_sf1_2_3': 'Exponential 5000',
    'gate_noise_exact_postzne_exponential_exact_sf1_2_3': 'Exponential no shots',
    'gate_noise_5k_postzne_richardson_5k_sf1_2_3': 'Richardson 5000 (after opt)',
    'gate_noise_exact_postzne_richardson_exact_sf1_2_3': 'Richardson no shots (after opt)',
    'gate_noise_5k_postzne_linear_5k_sf1_2_3': 'Linear 5000',
    'gate_noise_exact_postzne_linear_exact_sf1_2_3': 'Linear no shots',
    'gate_noise_5k_postpec_5k_s1000_global': 'PEC 5000',
    'gate_noise_exact_postpec_exact_s1000_global': 'PEC no shots',
    'fakeAthensV2_exact': 'AthensV2, no shots',
    'fakeAthensV2_5k': 'AthensV2, 5000 shots',
    'fakeAthensV2_exact_linear': 'Linear, no shots',
    'fakeAthensV2_5k_linear': 'Linear, 5000 shots',
    'fakeBogotaV2_exact': 'BogotaV2, no shots',
    'fakeBogotaV2_5k': 'BogotaV2, 5000 shots',
    'fakeManilaV2_exact': 'ManilaV2, no shots',
    'fakeManilaV2_5k': 'ManilaV2, 5000 shots',
    'gate_noise_default_5k': 'D2, 5k shots',
    'gate_noise_default_exact': 'D2, no shots',
    'gate_noise_real_5k': 'D3, 5k shots',
    'gate_noise_real_exact': 'D3, no shots',
    'gate_noise_zne_linear_5k_def': 'D2, Linear, 5k shots',
    'gate_noise_zne_linear_exact_def': 'D2, Linear, no shots',
    'gate_noise_zne_linear_5k_real': 'D3, Linear, 5k shots',
    'gate_noise_zne_linear_exact_real': 'D3, Linear, no shots',
    'gate_noise_zne_richardson_5k_def': 'D2, Richardson, 5k shots',
    'gate_noise_zne_richardson_exact_def': 'D2, Richardson, no shots',
    'gate_noise_zne_richardson_5k_real': 'D3, Richardson, 5k shots',
    'gate_noise_zne_richardson_exact_real': 'D3, Richardson, no shots',
    'gate_noise_zne_exponential_5k_def': 'D2, Exponential, 5k shots',
    'gate_noise_zne_exponential_exact_def': 'D2, Exponential, no shots',
    'gate_noise_zne_exponential_5k_real': 'D3, Exponential, 5k shots',
    'gate_noise_zne_exponential_exact_real': 'D3, Exponential, no shots',

    # ... add all your cost functions
}

optimizer_labels = {
    'mpi_hopso': 'HOPSO',
    'mpi_pso': 'PSO',
    'de': 'DE',
    'cobyla': 'COBYLA',
    'pso': 'PSO'
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
        y_label_map = optimizer_labels
        order = custom_cf
    else:
        custom = custom_cf
        attr = 'cost_function'
        x = 'optimizer'
        x_label_map = optimizer_labels
        y_label_map = cost_function_labels
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
        #ax1.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
        ax1.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
        ax1.legend()

        #sns.boxplot(data=subset, x='optimizer', y='real_energy', order=optimizers, width=0.2, showfliers=False, ax=ax2)
        #ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

        plt.title(f"{y_label_map[i]}")
        plt.ylabel("Energy (Hartree)")
        plt.xlabel("Depolarizing noise type, ZNE method, shots")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Sanitize filename
        out_path = os.path.join(OUTPUT_DIR, f"boxplot_D3_zne_{i}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

def check_type(str, type):
    if type in str :
        return True
    else: 
        False

def custom_multiple(setting):
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

    subset = combined[combined['optimizer'] == 'mpi_hopso']
    subset1 = subset[subset['cost_function'].str.contains('linear')]
    subset2 = subset[subset['cost_function'].str.contains('richardson')]
    #subset3 = subset[subset['cost_function'].str.contains('exponential')]
    melted_1 = subset1.melt(
        id_vars=[x],
        value_vars=['final_energy', 'real_energy'],
        var_name='energy_type',
        value_name='energy'
    )
    melted_2 = subset2.melt(
        id_vars=[x],
        value_vars=['final_energy', 'real_energy'],
        var_name='energy_type',
        value_name='energy'
    )

    melted_1[x] = melted_1[x].map(x_label_map)
    melted_1['energy_type'] = melted_1['energy_type'].map(energy_type_labels)
    melted_2[x] = melted_2[x].map(x_label_map)
    melted_2['energy_type'] = melted_2['energy_type'].map(energy_type_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

    labeled_order = [x_label_map[raw] for raw in order]
    present_in_1 = [cat for cat in labeled_order if cat in melted_1[x].unique()]
    present_in_2 = [cat for cat in labeled_order if cat in melted_2[x].unique()]

    color_dict = {'Reported Energy': 'steelblue', 'Actual Energy': 'darkorange'}
    sns.boxplot(data=melted_1, x=x, y='energy', hue='energy_type', order=present_in_1, width=0.2, showfliers=False, ax=ax1, palette=color_dict)
    sns.boxplot(data=melted_2, x=x, y='energy', hue='energy_type', order=present_in_2, width=0.2, showfliers=False, ax=ax2, palette=color_dict)
    #sns.boxplot(data=melted_3, x=x, y='energy', hue='energy_type', order=labeled_order, width=0.2, showfliers=False, ax=ax3, palette=color_dict)
    ax1.set_title('Linear ZNE')
    ax2.set_title('Richardson ZNE')
    
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    #ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax1.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
    ax1.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
    ax1.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
    ax2.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
    ax2.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
    ax2.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
    #ax3.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
    #ax3.axhline(y=E_EXACT - PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=None)
    #ax3.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
    ax1.legend()
    ax2.legend()
    #ax3.legend()
    ax1.xaxis.set_tick_params(rotation=45)
    ax2.xaxis.set_tick_params(rotation=45)
    #ax3.xaxis.set_tick_params(rotation=45)

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    fig.supylabel("Energy (Hartree)")          
    fig.supxlabel("Method and Shots")

    ax1.set_xlabel('')
    ax2.set_xlabel('')
    #ax3.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    #ax3.set_ylabel('')

    #sns.boxplot(data=subset, x='optimizer', y='real_energy', order=optimizers, width=0.2, showfliers=False, ax=ax2)
    #ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

    # Sanitize filename
    out_path = os.path.join(OUTPUT_DIR, f"boxplot_mitiq.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


custom(setting)
#per_cf()
#per_optimizer()
#custom_multiple(setting)
print("All box plots generated.")
