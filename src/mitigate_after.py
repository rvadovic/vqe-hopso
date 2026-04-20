"""
mitigate_after.py
========================
Post-hoc error mitigation (ZNE or PEC) applied to the best-found positions
stored in a HOPSO result CSV file.

Usage examples
--------------
# ZNE with linear extrapolation, 5 000 shots per scaled circuit
python mitigate_hopso_result.py \
    --input results/mpi_hopso_gate_noise_5k.csv \
    --method zne --zne_method linear --shots 5000

# ZNE with Richardson extrapolation, exact (statevector) simulation
python mitigate_hopso_result.py \
    --input results/mpi_hopso_gate_noise_5k.csv \
    --method zne --zne_method richardson --shots 0

# PEC with 5 000 shots and 500 quasi-probability samples
python mitigate_hopso_result.py \
    --input results/mpi_hopso_gate_noise_5k.csv \
    --method pec --shots 5000 --pec_samples 500

CLI arguments
-------------
--input         Path to the input CSV file (must contain a `best_position` column).
--method        Mitigation method: `zne` or `pec`.
--zne_method    ZNE extrapolation: `linear`, `richardson`, or `exponential`.
                Only used when --method=zne.  Default: linear.
--shots         Shots per circuit execution.  0 means exact/statevector.
                Default: 5000.
--pec_samples   Number of PEC quasi-probability samples.  Default: 200.
--scale_factors Comma-separated noise scale factors for ZNE.
                Default: 1.0,2.0,3.0
--output        Explicit output CSV path.  Auto-generated from input name if omitted.

Output CSV columns
------------------
run             Original run index.
final_energy    Mitigated energy (replaces original noisy energy).
time            Original optimisation time + mitigation wall-clock time (seconds).
best_position   Unchanged best-found parameter vector from the input file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time as _time
import warnings
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Qiskit / Aer / mitiq imports
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import Estimator
from mitiq import pec
from mitiq.pec.representations import represent_operation_with_local_depolarizing_noise
from mitiq.pec.representations import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_global_depolarizing_noise
)

from mitiq.pec import sample_circuit

import src.costF.costF_4q_H2_qiskit as q





def apply_zne(
    angles: np.ndarray,
    method: str = "linear",
    shots: Optional[int] = 5000,
    scale_factors: Optional[List[float]] = None,
) -> float:
    """
    Apply Zero-Noise Extrapolation to a single parameter vector.

    Strategy: build estimators with scaled depolarising probabilities
    (p1 * factor, p2 * factor) and evaluate <H> at each scale, then
    extrapolate to factor = 0.

    Parameters
    ----------
    angles        : 1-D array of ansatz parameters (length = ansatz.num_parameters).
    method        : Extrapolation method — 'linear', 'richardson', 'exponential'.
    shots         : Shots per scaled-noise circuit.  None → exact statevector.
    scale_factors : Noise-scale factors to use.  Default [1.0, 2.0, 3.0].

    Returns
    -------
    Extrapolated energy (float).
    """
    if method == 'linear':
        zne = q.gate_noise_zne_linear_exact
    elif method == 'richardson':
        zne = q.gate_noise_zne_richardson_exact
    return zne(angles)


def _qubit_count_of_rep(rep) -> int:
    """
    Return the number of qubits the ideal circuit of an OperationRepresentation acts on.
 
    mitiq stores `rep.ideal` as the same QPROGRAM type that was passed in
    (Qiskit QuantumCircuit here), so `num_qubits` is the Qiskit attribute.
    We fall back to Cirq's `all_qubits()` in case mitiq has converted it
    internally.
    """
    ideal = rep.ideal
    if hasattr(ideal, "num_qubits"):          # Qiskit QuantumCircuit
        return int(ideal.num_qubits)
    if hasattr(ideal, "all_qubits"):          # Cirq Circuit
        return len(list(ideal.all_qubits()))
    raise AttributeError(f"Cannot determine qubit count for {type(ideal)}")

# ===========================================================================
# PEC — two-noise-level representation builder + execution
# ===========================================================================
def _build_pec_representations(circuit: QuantumCircuit) -> list:
    """
    Build PEC quasi-probability representations for a circuit that has
    TWO distinct depolarising noise levels — one for single-qubit gates
    (DEPOL_PROB_1Q = 0.01) and one for two-qubit gates (DEPOL_PROB_2Q = 0.02).
 
    Why the mini-circuit approach does NOT work
    -------------------------------------------
    mitiq matches representations to circuit operations using structural
    circuit equality: it converts every operation in the circuit to a small
    "operation circuit" via its internal converter and then looks that circuit
    up in the provided representation list.  If we build independent mini
    QuantumCircuit objects outside that conversion pipeline, the qubit labels
    and internal gate objects produced by mitiq's converter don't match our
    hand-crafted circuits, which is exactly what caused the
    "No representation found" warnings.
 
    The correct approach
    --------------------
    We must let `represent_operations_in_circuit_with_local_depolarizing_noise`
    extract operations from the *same* full bound circuit that we will later
    pass to `execute_with_pec`.  That guarantees the ideal circuits inside the
    representations come from the same conversion pipeline and will match.
 
    We call the function **twice** — once with DEPOL_PROB_1Q and once with
    DEPOL_PROB_2Q — and then select by qubit count:
        • 1-qubit gate representations  →  keep those built with DEPOL_PROB_1Q
        • 2-qubit gate representations  →  keep those built with DEPOL_PROB_2Q
 
    Parameters
    ----------
    circuit : Qiskit QuantumCircuit with ALL parameters already bound (no
              free ParameterVector symbols remaining).
 
    Returns
    -------
    List of mitiq OperationRepresentation objects ready for execute_with_pec.
    """
    # Build representations for every unique operation in the circuit,
    # using each noise level in turn.  Both calls operate on the same
    # full circuit, so the ideal sub-circuits produced internally are
    # guaranteed to match what execute_with_pec will look for.
    reps_p1 = represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, q.depolarizing_prob1
    )
    reps_p2 = represent_operations_in_circuit_with_global_depolarizing_noise(
        circuit, q.depolarizing_prob2
    )
 
    selected: list = []
 
    # 1-qubit gates: use the p1 representation (correct noise level)
    for rep in reps_p1:
        try:
            if _qubit_count_of_rep(rep) == 1:
                selected.append(rep)
        except AttributeError as exc:
            # Cannot determine arity — include conservatively and warn.
            print(f"    [PEC] Warning: could not determine qubit count "
                  f"({exc}); including with p1 representation.")
            selected.append(rep)
 
    # 2-qubit gates: use the p2 representation (correct noise level)
    for rep in reps_p2:
        try:
            if _qubit_count_of_rep(rep) >= 2:
                selected.append(rep)
        except AttributeError as exc:
            print(f"    [PEC] Warning: could not determine qubit count "
                  f"({exc}); skipping p2 candidate.")
 
    if not selected:
        raise RuntimeError(
            "No PEC representations could be selected.  "
            "Ensure the circuit is fully bound (no free parameters) and "
            "contains at least one representable gate."
        )
 
    return selected


def apply_pec(
    angles: np.ndarray,
    shots: Optional[int] = 5000,
    num_samples: int = 200,
) -> float:
    """
    Apply Probabilistic Error Cancellation to a single parameter vector.

    The function:
      1. Binds `angles` into the ansatz circuit.
      2. Builds per-gate-type OperationRepresentations (1-q vs 2-q noise).
      3. Calls mitiq's `execute_with_pec` to run sampled quasi-circuits
         under the two-depolarising-channel noise model.

    Parameters
    ----------
    angles      : 1-D array of ansatz parameters.
    shots       : Shots per sampled circuit.  None → exact statevector.
    num_samples : Number of quasi-probability samples drawn by PEC.
                  Higher = lower variance, more expensive.
                  Typical range: 100–2000.

    Returns
    -------
    Mitigated energy expectation value (float).
    """
    circuit = q.ansatz.assign_parameters(angles)
    executor = q.execute_exact
    representations = _build_pec_representations(circuit)

    if not representations:
        raise RuntimeError("No PEC representations could be built for this circuit. "
                           "Check that the circuit contains supported gate types.")

    mitigated = pec.execute_with_pec(
        circuit,
        executor,
        representations=representations,
        num_samples=num_samples,
    )
    return float(mitigated)


# ===========================================================================
# CSV helpers  (mirrors result_handler_csv.py logic)
# ===========================================================================
def _parse_position(val) -> Optional[list]:
    """Parse a best_position cell — handles JSON (new) and ast (old) formats."""
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(val)
        except Exception:
            return None


def _read_csv(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as fh:
        lines = fh.readlines()
    return pd.read_csv(StringIO("".join(lines)), sep=";")


def _write_csv(out_path: str, rows: list) -> None:
    if not rows:
        print("Warning: no rows to write.", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Output path generation
# ===========================================================================
def _auto_output_path(input_path: str, method: str, zne_method: str,
                       shots: Optional[int], pec_samples: int,
                       scale_factors: List[float]) -> str:
    results_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)),)
    base = os.path.splitext(os.path.basename(input_path))[0]
    shots_tag = "exact" if shots is None else f"{shots // 1000}k"

    if method == "zne":
        sf_tag = "_".join(str(f).replace(".", "") for f in scale_factors)
        tag = f"postzne_{zne_method}_{shots_tag}_sf{sf_tag}"
    else:
        tag = f"postpec_{shots_tag}_s{pec_samples}"

    return os.path.join(results_dir, f"{base}_{tag}.csv")


# ===========================================================================
# Main
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Apply ZNE or PEC post-hoc to HOPSO best-found positions.\n"
            "Reads a results CSV, mitigates each row's best_position, and\n"
            "writes a new CSV with updated energies and cumulative times."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="Path to input CSV (must contain 'best_position' column)")
    p.add_argument("--method",     required=True, choices=["zne", "pec"],
                   help="Mitigation method")
    p.add_argument("--zne_method", default="linear",
                   choices=["linear", "richardson", "exponential"],
                   help="ZNE extrapolation method (only for --method=zne)")
    p.add_argument("--shots",      type=int, default=5000,
                   help="Shots per circuit; 0 = exact statevector (default: 5000)")
    p.add_argument("--pec_samples", type=int, default=200,
                   help="PEC quasi-probability samples (default: 200)")
    p.add_argument("--scale_factors", default="1.0,2.0,3.0",
                   help="Comma-separated ZNE scale factors (default: 1.0,2.0,3.0)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (auto-generated if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    shots: Optional[int] = None if args.shots == 0 else args.shots
    scale_factors: List[float] = [float(x) for x in args.scale_factors.split(",")]

    # ── Read input ────────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        sys.exit(f"Error: input file not found: {args.input}")

    df = _read_csv(args.input)

    if "best_position" not in df.columns:
        sys.exit(
            "Error: the input CSV does not contain a 'best_position' column.\n"
            "Only result files that recorded the best-found parameter vector\n"
            "are supported.  Re-run the HOPSO experiment with position saving\n"
            "enabled, or choose a different result file."
        )

    # ── Determine output path ─────────────────────────────────────────────
    out_path = args.output or _auto_output_path(
        args.input, args.method, args.zne_method,
        shots, args.pec_samples, scale_factors,
    )

    # ── Banner ────────────────────────────────────────────────────────────
    shots_label = "exact (statevector)" if shots is None else f"{shots:,} shots"
    print("=" * 60)
    print("  Post-hoc error mitigation for HOPSO results")
    print("=" * 60)
    print(f"  Input      : {args.input}")
    print(f"  Method     : {args.method.upper()}"
          + (f"  ({args.zne_method})" if args.method == "zne" else ""))
    if args.method == "zne":
        print(f"  Scale facs : {scale_factors}")
    else:
        print(f"  PEC samples: {args.pec_samples}")
    print(f"  Shots      : {shots_label}")
    print(f"  Rows       : {len(df)}")
    print(f"  Output     : {out_path}")
    print("=" * 60)

    # ── Process each run ──────────────────────────────────────────────────
    output_rows = []
    n_skipped = 0
    q.prepare_estimators_zne_exact()
    for idx, row in df.iterrows():
        run_id = int(row.get("run", idx + 1))
        position = _parse_position(row["best_position"])

        if position is None:
            print(f"  Run {run_id:3d}: no position found — skipped.")
            n_skipped += 1
            continue

        angles = np.array(position, dtype=float)
        orig_time = float(row["time"])
        orig_energy = float(row["final_energy"])

        print(f"  Run {run_id:3d}: original energy = {orig_energy:+.6f} …", end="  ", flush=True)
        t0 = _time.perf_counter()

        try:
            if args.method == "zne":
                mitigated_energy = apply_zne(
                    angles,
                    method=args.zne_method,
                    shots=shots,
                    scale_factors=scale_factors,
                )
            else:  # pec
                mitigated_energy = apply_pec(
                    angles,
                    shots=shots,
                    num_samples=args.pec_samples,
                )
        except Exception as exc:
            print(f"ERROR — {exc}")
            n_skipped += 1
            continue

        mit_time = _time.perf_counter() - t0
        total_time = orig_time + mit_time

        improvement = orig_energy - mitigated_energy   # negative = got worse
        direction   = "↓" if improvement > 0 else "↑"
        print(
            f"mitigated = {mitigated_energy:+.6f}  "
            f"({direction}{abs(improvement):.4f})  "
            f"+{mit_time:.1f}s"
        )

        output_rows.append({
            "run":           run_id,
            "final_energy":  mitigated_energy,
            "time":          round(total_time, 6),
            "best_position": row["best_position"],   # preserve original
        })

    # ── Write output ──────────────────────────────────────────────────────
    _write_csv(out_path, output_rows)

    print()
    print(f"Wrote {len(output_rows)} mitigated rows  ({n_skipped} skipped).")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()