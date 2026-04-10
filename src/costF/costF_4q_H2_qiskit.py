from qiskit.quantum_info import SparsePauliOp
import numpy as np
from numpy import linalg as LA
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit import QuantumCircuit,ParameterVector
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, ReadoutError
from qiskit_aer import AerSimulator
from mitiq import zne, pec
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.pec.representations import represent_operations_in_circuit_with_local_depolarizing_noise
from scipy.optimize import curve_fit
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeAthensV2, FakeBogotaV2
from qiskit import transpile
from qiskit.transpiler import Layout
#from tqdm import tqdm

#4-qubit labels
labels = ["IIII","ZIII","ZZII","IIZI","IZZZ","IZII","ZIZI","XZXI","XIXZ","XIXI","XZXZ","ZZZZ","ZZZI","ZIZZ","IZIZ"]
coeffs = [-0.80718, 0.17374, -0.23047, 0.17374, -0.23047,  0.12149, 0.1694, -0.04509, 0.04509, 0.04509, -0.04509,  0.16658,  0.16658, 0.17511, 0.12149]

# Hamiltonian
H = SparsePauliOp.from_list(list(zip(labels, coeffs)))

eigvals = LA.eig(H.to_matrix())
E_exact = np.min(eigvals[0])

# Ansatz
ansatz = TwoLocal(4, ["ry"],"cx", reps=3, entanglement="linear",insert_barriers=True).decompose()

# Fake backends
fake_backend = FakeBogotaV2()

noise_model_fakeManilaV2 = NoiseModel.from_backend(fake_backend)
noise_model_fakeAthensV2 = NoiseModel.from_backend(fake_backend)
noise_model_fakeBogotaV2 = NoiseModel.from_backend(fake_backend)

I = SparsePauliOp(["I"], coeffs=[1.0])
H_5q = H.tensor(I)

transpiled_ansatz = transpile(ansatz, backend=fake_backend, initial_layout=[0, 1, 2 ,3])

estimator_fakeManilaV2 = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_fakeManilaV2,
            "method": "automatic"
        },
        "run_options": {
            "shots": 5000
        }
    }
)

estimator_fakeAthensV2 = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_fakeAthensV2,
            "method": "automatic"
        },
        "run_options": {
            "shots": 5000
        }
    }
)

estimator_fakeBogotaV2 = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_fakeBogotaV2,
            "method": "automatic"
        },
        "run_options": {
            "shots": 5000
        }
    }
)

# Noiseless estimator
estimator_noiseless = EstimatorV2()

# Shot noise estimators

noise_model_shots = NoiseModel()

estimator_shot_noise_5k = EstimatorV2(
    options={
        "default_precision": 1/np.sqrt(5000),
    }
)

estimator_shot_noise_1k = EstimatorV2(
    options={
        "default_precision": 1/np.sqrt(1000),
    }
)

estimator_shot_noise_100 = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_shots,
            "method": "automatic"
        },
        "run_options": {
            "shots": 100
        }
    }
)

estimator_shot_noise_100k = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_shots,
            "method": "automatic"
        },
        "run_options": {
            "shots": 100000
        }
    }
)

#Gate noise estimator
depolarizing_prob1 = 0.01
depolarizing_prob2 = 0.02
error1 = depolarizing_error(depolarizing_prob1, 1)
error2 = depolarizing_error(depolarizing_prob2, 2)

noise_model_depolarization = NoiseModel()
noise_model_depolarization.add_all_qubit_quantum_error(error1, ['r'])
noise_model_depolarization.add_all_qubit_quantum_error(error2, ['cx'])

#readout_prob = 0.02
#readout = ReadoutError([[1 - readout_prob, readout_prob], [readout_prob, 1 - readout_prob]])

#noise_model_depolarization.add_all_qubit_readout_error(readout) # Not yet

estimator_gate_noise_5k = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_depolarization,
            "method": "automatic"
        },
        "run_options": {
            "shots": 5000
        }
    }
)

estimator_gate_noise_100k = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_depolarization,
            "method": "automatic"
        },
        "run_options": {
            "shots": 100000
        }
    }
)

estimator_gate_noise_1k = EstimatorV2(
    options={
        "backend_options": {
            "noise_model": noise_model_depolarization,
            "method": "automatic"
        },
        "run_options": {
            "shots": 1000
        }
    }
)

def exp_function(x, a, b, c):
    return a * np.exp(b * x) - c

def extrapolate(scale_factors, energies, method):
    # energies shape: (len(scale_factors),) for a single parameter set
    if method == 'linear':
        coeffs = np.polyfit(scale_factors, energies, 1)
        return coeffs[1]  # intercept at 0
    elif method == 'richardson':
        if len(scale_factors) == 3:
            E1, E2, E3 = energies
            return ((9*E1) - (8*E3) + E2) / (2)
    elif method == 'exponential':
        try:
            p0 = [1.2, 0.5, E_exact + 1.0]  # Initial guess: a, b, c
            popt, _ = curve_fit(exp_function, scale_factors, energies, p0=p0, maxfev=1000)
            return popt[0] - popt[2]  # a + c at x=0
        except (RuntimeError, RuntimeWarning):
            print("Exponential fit failed, falling back to linear extrapolation")
            coeffs = np.polyfit(scale_factors, energies, 1)
            return coeffs[1]  # intercept at 0
        
def calc_probability(prob, factor):
    return min(1.0, prob * factor)

# SCALE FACTORS FOR ZNE
scale_factors = [1.0, 2.0, 3.0]

estimators_zne = []

def prepare_estimators_zne_5k():
    for factor in scale_factors:
        scaled_error1 = depolarizing_error(calc_probability(depolarizing_prob1, factor), 1)
        scaled_error2 = depolarizing_error(calc_probability(depolarizing_prob2, factor), 2)

        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(scaled_error1, ['r'])
        scaled_noise.add_all_qubit_quantum_error(scaled_error2, ['cx'])

        #readout_prob_scaled = calc_probability(readout_prob, factor)
        #scaled_readout = ReadoutError([[1 - readout_prob_scaled, readout_prob_scaled], [readout_prob_scaled, 1 - readout_prob_scaled]])

        #scaled_noise.add_all_qubit_readout_error(scaled_readout)

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

def prepare_estimators_zne_100k():
    for factor in scale_factors:
        scaled_error1 = depolarizing_error(calc_probability(depolarizing_prob1, factor), 1)
        scaled_error2 = depolarizing_error(calc_probability(depolarizing_prob2, factor), 2)

        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(scaled_error1, ['r'])
        scaled_noise.add_all_qubit_quantum_error(scaled_error2, ['cx'])

        #readout_prob_scaled = calc_probability(readout_prob, factor)
        #scaled_readout = ReadoutError([[1 - readout_prob_scaled, readout_prob_scaled], [readout_prob_scaled, 1 - readout_prob_scaled]])

        #scaled_noise.add_all_qubit_readout_error(scaled_readout)

        estimator = EstimatorV2(
            options={
                "backend_options": {
                    "noise_model": scaled_noise,
                    "method": "automatic"
                },
                "run_options": {
                    "shots": 100000
                }
            }
        )
        estimators_zne.append(estimator)

def gate_noise_zne_richardson_5k(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        #print(f"Energies 1, 2, 3: {raw_energies}")
        # extrapolate
        mit = extrapolate(scale_factors, raw_energies, method='richardson')
        mitigated.append(mit)
    return np.array(mitigated)

def gate_noise_zne_linear_5k(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        #print(f"Energies 1, 2, 3: {raw_energies}")
        # extrapolate
        mit = extrapolate(scale_factors, raw_energies, method='linear')
        mitigated.append(mit)
    return np.array(mitigated)

def gate_noise_zne_exponential_5k(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        #print(f"Energies 1, 2, 3: {raw_energies}")
        # extrapolate
        mit = extrapolate(scale_factors, raw_energies, method='exponential')
        mitigated.append(mit)
    return np.array(mitigated)

def gate_noise_zne_richardson_100k(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        #print(f"Energies 1, 2, 3: {raw_energies}")
        # extrapolate
        mit = extrapolate(scale_factors, raw_energies, method='richardson')
        mitigated.append(mit)
    return np.array(mitigated)

def gate_noise_zne_linear_100k(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        #print(f"Energies 1, 2, 3: {raw_energies}")
        # extrapolate
        mit = extrapolate(scale_factors, raw_energies, method='linear')
        mitigated.append(mit)
    return np.array(mitigated)

def gate_noise_zne_exponential_100k(angles):
    mitigated = []
    for a in angles:
        raw_energies = []
        for est in estimators_zne:
            bound = ansatz.assign_parameters(a)
            pubs = [(bound, H)]
            job = est.run(pubs)
            result = job.result()
            raw_energies.append(result[0].data.evs)
        #print(f"Energies 1, 2, 3: {raw_energies}")
        # extrapolate
        mit = extrapolate(scale_factors, raw_energies, method='exponential')
        mitigated.append(mit)
    return np.array(mitigated)

def execute_5k(circuit):
    pubs = [(circuit, H)]
    job = estimator_gate_noise_5k.run(pubs)
    result = job.result()
    energy = result[0].data.evs
    return energy

def execute_100k(circuit):
    pubs = [(circuit, H)]
    job = estimator_gate_noise_100k.run(pubs)
    result = job.result()
    energy = result[0].data.evs
    return energy

def mitiq_extrapolate(angles, scale_factors, method, shots):
    if method == 'linear':
        factory = zne.inference.LinearFactory(scale_factors)
    elif method == 'richardson':
        factory = zne.inference.RichardsonFactory(scale_factors)
    elif method == 'exponential':
        factory = zne.inference.ExponentialFactory(scale_factors)

    extrapolation_method = factory.extrapolate
    mitigated = []

    for a in angles:
        circuit = ansatz.assign_parameters(a)
        folded_circuits = zne.construct_circuits(circuit=circuit, scale_factors=scale_factors, scale_method=fold_gates_at_random)
        if shots == 5000:
            energies = [execute_5k(c) for c in folded_circuits]
        elif shots == 100000:
            energies = [execute_100k(c) for c in folded_circuits]
        mit = zne.combine_results(scale_factors, energies, extrapolation_method)
        mitigated.append(mit)

    return np.array(mitigated)

def gate_noise_zne_mitiq_linear_5k(angles):
    return mitiq_extrapolate(angles, scale_factors, method='linear', shots=5000)

def gate_noise_zne_mitiq_richardson_5k(angles):
    return mitiq_extrapolate(angles, scale_factors, method='richardson', shots=5000)

def gate_noise_zne_mitiq_exponential_5k(angles):
    return mitiq_extrapolate(angles, scale_factors, method='exponential', shots=5000)

def gate_noise_zne_mitiq_linear_100k(angles):
    return mitiq_extrapolate(angles, scale_factors, method='linear', shots=100000)

def gate_noise_zne_mitiq_richardson_100k(angles):
    return mitiq_extrapolate(angles, scale_factors, method='richardson', shots=100000)

def gate_noise_zne_mitiq_exponential_100k(angles):
    return mitiq_extrapolate(angles, scale_factors, method='exponential', shots=100000)

def gate_noise_pec_mitiq_5k(angles, last_iteration):
    if last_iteration:
        mitigated = []
        for a in angles:
            # extrapolate
            circuit = ansatz.assign_parameters(a)
            reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level=depolarizing_prob2)
            mit = pec.execute_with_pec(circuit, execute_5k, representations=reps, num_samples=1000)
            mitigated.append(mit)
        return np.array(mitigated)
    else:
        return gate_noise_5k(angles)

def gate_noise_pec_mitiq_100k(angles, last_iteration):
    if last_iteration:
        mitigated = []
        for a in angles:
            # extrapolate
            circuit = ansatz.assign_parameters(a)
            reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level=depolarizing_prob2)
            mit = pec.execute_with_pec(circuit, execute_100k, representations=reps, num_samples=1000)
            mitigated.append(mit)
        return np.array(mitigated)
    else:
        return gate_noise_100k(angles)

def noiseless(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_noiseless.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def shot_noise_5k(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_shot_noise_5k.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def shot_noise_1k(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_noiseless.run(pubs, precision=1/np.sqrt(1000))
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def shot_noise_100k(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_shot_noise_100k.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def shot_noise_100(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_shot_noise_100.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def gate_noise_5k(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_gate_noise_5k.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def gate_noise_100k(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_gate_noise_100k.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def gate_noise_1k(angles):
    bound_circuits = [ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H) for c in bound_circuits]
    job = estimator_gate_noise_1k.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def fakeManilaV2(angles):
    bound_circuits = [transpiled_ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H_5q) for c in bound_circuits]
    job = estimator_fakeManilaV2.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def fakeAthensV2(angles):
    bound_circuits = [transpiled_ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H_5q) for c in bound_circuits]
    job = estimator_fakeAthensV2.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies

def fakeBogotaV2(angles):
    bound_circuits = [transpiled_ansatz.assign_parameters(a) for a in angles]
    pubs = [(c, H_5q) for c in bound_circuits]
    job = estimator_fakeBogotaV2.run(pubs)
    result = job.result() 
    energies = np.array([res.data.evs for res in result])
    return energies