import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from costF.costF_4q_H2_qiskit import E_exact

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "graphs")
PRECISION = 1.59e-3
E_EXACT = E_exact.real
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Load all raw data into a single DataFrame
all_data = []
for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.csv'):
        continue
    filepath = os.path.join(DATA_DIR, fname)
    try:
        df = pd.read_csv(filepath)
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
    except:
        print(f"Warning: could not parse {fname} – skipping")
        continue

    # Keep only needed columns
    temp = df[['final_energy']].copy()
    temp['optimizer'] = optimizer
    temp['cost_function'] = costF
    all_data.append(temp)

if not all_data:
    raise RuntimeError("No valid raw data files found.")

combined = pd.concat(all_data, ignore_index=True)


optimizers = sorted(combined['optimizer'].unique())
cost_functions = sorted(combined['cost_function'].unique())

# -------------------------------
# 1. Box plots: per cost function (one figure per cost function)
# -------------------------------
for cf in cost_functions:
    subset = combined[combined['cost_function'] == cf]
    if subset.empty:
        continue

    plt.figure(figsize=(8, 6))

    ax = sns.boxplot(data=subset, x='optimizer', y='final_energy', order=optimizers, width=0.2, showfliers=False)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
    ax.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
    ax.legend()
    
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
for opt in optimizers:
    subset = combined[combined['optimizer'] == opt]
    if subset.empty:
        continue

    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(data=subset, x='cost_function', y='final_energy', order=cost_functions, width=0.2, showfliers=False)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.axhline(y=E_EXACT, color='green', linestyle='-', linewidth=1.5, label=f'Exact: {E_EXACT:.4f}')
    ax.axhline(y=E_EXACT + PRECISION, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Chem. acc. (±{PRECISION:.4f})')
    ax.legend()

    plt.title(f"Final Energy Distribution – Optimizer: {opt}")
    plt.ylabel("Energy (Hartree)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"boxplot_optimizer_{opt}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

print("All box plots generated.")