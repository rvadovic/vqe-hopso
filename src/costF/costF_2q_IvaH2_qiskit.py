

# from file: 2qh2.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:45:34 2024

@author: mirac
"""

from qiskit.quantum_info import SparsePauliOp
import numpy as np

#2-qubit labels
labels = ["II","ZI","IZ","ZZ","XX"]
coeffs = [-1.05016,0.40421,0.40421,0.01135,0.18038]
H = SparsePauliOp(labels,coeffs)

from numpy import linalg as LA
eig = LA.eig(H.to_matrix())

#from tqdm import tqdm
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit import QuantumCircuit,ParameterVector
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError
from qiskit_aer import AerSimulator

# Noiseless estimator
sim_noiseless = AerSimulator()
estimator_noiseless = EstimatorV2.from_backend(sim_noiseless)

# Shot noise estimator
estimator_shot_noise = EstimatorV2.from_backend(sim_noiseless, run_options={"shots": 5000})

#Gate noise estimator
error1 = depolarizing_error(0.01, 1)
error2 = depolarizing_error(0.02, 2)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['r'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

readout = ReadoutError([[0.98,0.02],
                        [0.02,0.98]])

noise_model.add_all_qubit_readout_error(readout)

sim = AerSimulator(noise_model = noise_model, method = 'automatic')
estimator = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model,
            "method": "automatic"
        },
        "run_options": {
            "shots": 5000
        }
    }
)

ansatz = TwoLocal(2, ["ry"],"cx", reps=2, entanglement="linear",insert_barriers=True).decompose()

def cost_function_noiseless(angles):
    pubs = [(ansatz, H, angle) for angle in angles]
    job = estimator_noiseless.run(pubs)
    result = job.result() # It will block until the job finishes.
    energies = np.array([res.data.evs for res in result])
    return energies