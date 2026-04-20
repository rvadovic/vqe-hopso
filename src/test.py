"""
pec_sanity_check.py
-------------------
Verifies PEC self-consistency on the simplest possible case:
  - 1 qubit, 1 Ry gate
  - Observable: Z  (so <Z> = cos(theta))
  - Noise: single-qubit depolarising with p = 0.01

Expected behaviour:
  noiseless <Z>  =  cos(0.7)  ≈  0.7648
  noisy <Z>      =  (1 - 4p/3) * cos(0.7)  ≈  0.7546   [for p=0.01]
  PEC estimate   →  cos(0.7)  ≈  0.7648  (as num_samples grows)

If PEC mean drifts significantly BELOW 0.7648, the noise level in the
representations is over-estimated relative to what the executor applies.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import pec
from mitiq.pec.representations import (
    represent_operations_in_circuit_with_local_depolarizing_noise,
    represent_operations_in_circuit_with_global_depolarizing_noise
)

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

# ── 0. Parameters ────────────────────────────────────────────────────────────
THETA       = 0.7
DEPOL_PROB  = 0.01
NUM_SAMPLES = 2000   # increase to 5000+ for tighter convergence
SHOTS       = None   # None = exact statevector (isolates PEC itself from shot noise)

# Minimal two-noise-level circuit
qc = QuantumCircuit(2)
qc.ry(0.7, 0)
qc.cx(0, 1)

# Noise model with two levels
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["r"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])

# Observable: ZZ
observable = SparsePauliOp(["ZZ"], coeffs=[1.0])

# ── 3. Noiseless reference ───────────────────────────────────────────────────
estimator_noiseless = Estimator(
    run_options={"shots": None},
    approximation=True
)
job = estimator_noiseless.run([qc], [observable])
noiseless_value = job.result().values[0]
print(f"Noiseless  <Z> = {noiseless_value:.6f}  (expected: {np.cos(0.35) + np.sin(0.35):.6f})")

# ── 4. Noisy reference ───────────────────────────────────────────────────────
# This is what we see WITHOUT any mitigation.

estimator_noisy = Estimator(
    backend_options={"noise_model": noise_model},
    run_options={"shots": SHOTS},
    approximation=(SHOTS is None)
)
job = estimator_noisy.run([qc], [observable])
noisy_value = job.result().values[0]
# Analytical prediction: depolarising shrinks Bloch vector by (1 - 4p/3)
predicted_noisy = (1 - 4 * DEPOL_PROB / 3) * np.cos(THETA)
print(f"Noisy      <Z> = {noisy_value:.6f}  (analytical prediction: {predicted_noisy:.6f})")

# ── 5. The executor for PEC ───────────────────────────────────────────────────
# CRITICAL: this executor must apply EXACTLY the same noise as the
# representations assume.  Here we use "ry" in both places.
def executor(circuit: QuantumCircuit) -> float:
    job = estimator_noisy.run([circuit], [observable])
    return float(job.result().values[0])

# ── 6. Build representations ──────────────────────────────────────────────────
# We pass the ideal_circuit so mitiq extracts operations via its own
# internal converter — guaranteeing the representations match what
# execute_with_pec will look for.
reps_p1 = represent_operations_in_circuit_with_local_depolarizing_noise(
        qc, 0.01
    )
reps_p2 = represent_operations_in_circuit_with_global_depolarizing_noise(
        qc, 0.02
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

print(f"\nNumber of representations built: {len(selected)}")
for r in selected:
    print(f"  {r}")

# ── 7. Run PEC ────────────────────────────────────────────────────────────────
pec_estimate = pec.execute_with_pec(
    qc,
    executor,
    representations=selected,
    num_samples=NUM_SAMPLES,
)
print(f"\nPEC        <Z> = {pec_estimate:.6f}")
print(f"Noiseless  <Z> = {noiseless_value:.6f}")
print(f"Difference     = {abs(pec_estimate - noiseless_value):.6f}")
print(f"\nDid PEC overcorrect below noiseless? {pec_estimate < noiseless_value - 0.01}")