from qiskit.quantum_info import SparsePauliOp
import numpy as np
from numpy import linalg as LA
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit import QuantumCircuit,ParameterVector
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError
from qiskit_aer import AerSimulator
#from tqdm import tqdm

#4-qubit labels
labels = ["IIII","ZIII","ZZII","IIZI","IZZZ","IZII","ZIZI","XZXI","XIXZ","XIXI","XZXZ","ZZZZ","ZZZI","ZIZZ","IZIZ"]
coeffs = [-0.80718, 0.17374, -0.23047, 0.17374, -0.23047,  0.12149, 0.1694, -0.04509, 0.04509, 0.04509, -0.04509,  0.16658,  0.16658, 0.17511, 0.12149]

H = SparsePauliOp.from_list(list(zip(labels, coeffs)))

eigvals = LA.eig(H.to_matrix())
E_exact = np.min(eigvals[0])

# Noiseless estimator
sim_noiseless = AerSimulator()
estimator_noiseless = EstimatorV2()

# Shot noise estimator
estimator_shot_noise = EstimatorV2(options={
        "run_options": {
            "shots": 5000
        }
    })

#Gate noise estimator
depolarizing_prob1 = 0.01
depolarizing_prob2 = 0.02
error1 = depolarizing_error(depolarizing_prob1, 1)
error2 = depolarizing_error(depolarizing_prob2, 2)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['r'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

readout_prob = 0.02
readout = ReadoutError([[1 - readout_prob, readout_prob], [readout_prob, 1 - readout_prob]])

noise_model.add_all_qubit_readout_error(readout)

#sim = AerSimulator(noise_model = noise_model, method = 'automatic')
estimator_gate_noise = EstimatorV2(
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

ansatz = TwoLocal(4, ["ry"],"cx", reps=3, entanglement="linear",insert_barriers=True).decompose()


import numpy as np
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

def extrapolate(noise_factors, energies, method ='richardson'):
    # energies shape: (len(noise_factors),) for a single parameter set
    if method == 'linear':
        coeffs = np.polyfit(noise_factors, energies, 1)
        return coeffs[1]  # intercept at 0
    elif method == 'richardson':
        if len(noise_factors) == 3:
            E1, E2, E3 = energies
            return (9*E1 - 8*E3 + E2) / (2)
        
def calc_probability(prob, factor):
    return min(1.0, prob * factor)

# Pre-create estimators for different noise factors
noise_factors = [1.0, 2.0, 3.0]
estimators_zne = []
for factor in noise_factors:
    scaled_error1 = depolarizing_error(calc_probability(depolarizing_prob1, factor), 1)
    scaled_error2 = depolarizing_error(calc_probability(depolarizing_prob2, factor), 2)

    scaled_noise = NoiseModel()
    scaled_noise.add_all_qubit_quantum_error(scaled_error1, ['r'])
    scaled_noise.add_all_qubit_quantum_error(scaled_error2, ['cx'])

    readout_prob_scaled = calc_probability(readout_prob, factor)
    scaled_readout = ReadoutError([[1 - readout_prob_scaled, readout_prob_scaled], [readout_prob_scaled, 1 - readout_prob_scaled]])

    scaled_noise.add_all_qubit_readout_error(scaled_readout)

    estimator = EstimatorV2(
        options={
            "backend_options": {
                "noise_model": scaled_noise,
                "method": "automatic"
            },
            "run_options": {
                "shots": 5000
            }
        }
    )
    estimators_zne.append(estimator)

def cost_function_gate_noise_zne(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        # extrapolate
        mit = extrapolate(noise_factors, raw_energies)
        mitigated.append(mit)
    return np.array(mitigated)

def cost_function_noiseless(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_noiseless.run(pubs)
    result = job.result() # It will block until the job finishes.
    energies = np.array([res.data.evs for res in result])
    return energies

def cost_function_shot_noise(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_shot_noise.run(pubs)
    result = job.result() # It will block until the job finishes.
    energies = np.array([res.data.evs for res in result])
    return energies

def cost_function_gate_noise(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_gate_noise.run(pubs)
    result = job.result() # It will block until the job finishes.
    energies = np.array([res.data.evs for res in result])
    return energies
