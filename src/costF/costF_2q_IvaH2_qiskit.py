

# from file: 2qh2.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:45:34 2024

@author: mirac
"""
#%%
from qiskit.quantum_info import SparsePauliOp
import numpy as np

#2-qubit labels
labels = ["II","ZI","IZ","ZZ","XX"]
coeffs = [-1.05016,0.40421,0.40421,0.01135,0.18038]
H = SparsePauliOp(labels,coeffs)
#%%
from numpy import linalg as LA
eig = LA.eig(H.to_matrix())

#%%
from tqdm import tqdm
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit,ParameterVector
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

error1 = depolarizing_error(0.05, 1)
error2 = depolarizing_error(0.1, 2)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['r'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

sim = AerSimulator(noise_model = noise_model, method = 'automatic')
estimator = Estimator(options = {"backend" : sim, "shots": 5000})

ansatz = TwoLocal(2, ["ry"],"cx", reps=2, entanglement="linear",insert_barriers=True).decompose()
#%%
def objective_function_1(angle):
    job = estimator.run([ansatz], [H], [angle])
    energy = job.result().values[0] # It will block until the job finishes.
    return energy