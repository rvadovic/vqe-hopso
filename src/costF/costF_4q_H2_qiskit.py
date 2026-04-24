from qiskit.quantum_info import SparsePauliOp
import numpy as np
from numpy import linalg as LA
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Estimator, EstimatorV2
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
fake_bogota = FakeBogotaV2()
fake_athens = FakeAthensV2()
fake_manila = FakeManilaV2()

noise_model_fakeManilaV2 = NoiseModel.from_backend(fake_manila)
noise_model_fakeAthensV2 = NoiseModel.from_backend(fake_athens)
noise_model_fakeBogotaV2 = NoiseModel.from_backend(fake_bogota)

I = SparsePauliOp(["I"], coeffs=[1.0])
H_5q = H.tensor(I)

transpiled_ansatz_bogota = transpile(ansatz, backend=fake_bogota, initial_layout=[0, 1, 2 ,3])
transpiled_ansatz_athens = transpile(ansatz, backend=fake_athens, initial_layout=[0, 1, 2 ,3])
transpiled_ansatz_manila = transpile(ansatz, backend=fake_manila, initial_layout=[0, 1, 2 ,3])

estimator_fakeManilaV2 = Estimator(
    backend_options={
        "noise_model": noise_model_fakeManilaV2
    },
    run_options={
        "shots": 5000
    }
)

estimator_fakeAthensV2 = Estimator(
    backend_options={
        "noise_model": noise_model_fakeAthensV2
    },
    run_options={
        "shots": 5000
    }
    
)

estimator_fakeBogotaV2 = Estimator(
    backend_options={
        "noise_model": noise_model_fakeBogotaV2
    },
    run_options={
        "shots": 5000
    }
    
)

estimator_fakeManilaV2_exact = Estimator(
    backend_options={
        "noise_model": noise_model_fakeManilaV2
    },
    run_options={
        "shots": None
    },
    approximation=True
)

estimator_fakeAthensV2_exact = Estimator(
    backend_options={
        "noise_model": noise_model_fakeAthensV2
    },
    run_options={
        "shots": None
    },
    approximation=True
    
)

estimator_fakeBogotaV2_exact = Estimator(
    backend_options={
        "noise_model": noise_model_fakeBogotaV2
    },
    run_options={
        "shots": None
    },
    approximation=True
    
)

# Noiseless estimator
estimator_noiseless = Estimator(
    run_options={
        "shots": None
    },
    approximation=True
)

estimator_shot_noise_5k = Estimator(
    run_options={
        "shots": 5000
    }
)

estimator_shot_noise_1k = Estimator(
    run_options={
        "shots": 1000
    }
)

estimator_shot_noise_100 = Estimator(
    run_options={
        "shots": 100
    },
    approximation=False
)

estimator_shot_noise_10k = Estimator(
    run_options={
        "shots": 10000
    }
)

estimator_shot_noise_100k = Estimator(
    run_options={
        "shots": 100000
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

estimator_gate_noise_exact = Estimator(
    backend_options={
        "noise_model": noise_model_depolarization
    },
    run_options={
        "shots": None
    },
    approximation=True
)

estimator_gate_noise_5k = Estimator(
    backend_options={
        "noise_model": noise_model_depolarization
    },
    run_options={
        "shots": 5000
    }
)

estimator_gate_noise_10k = Estimator(
    backend_options={
        "noise_model": noise_model_depolarization
    },
    run_options={
        "shots": 10000
    }
)

estimator_gate_noise_1k = Estimator(
    backend_options={
        "noise_model": noise_model_depolarization
    },
    run_options={
        "shots": 1000
    }
)

def exp_function(x, a, b):
    return a * np.exp(b * x)

def extrapolate(sf, energies, method):
    # energies shape: (len(scale_factors),) for a single parameter set
    if method == 'linear':
        coeffs = np.polyfit(sf, energies, 1)
        return coeffs[1]  # intercept at 0
    elif method == 'richardson':
        if len(sf) == 3:
            E1, E2, E3 = energies
            return ((3*E1) - (3*E2) + E3)
    elif method == 'exponential':
        try:
            p0 = [-1.46985, 0.923858]  # Initial guess: a, b
            b0 = np.log(energies[1] / energies[0])   # ratio of adjacent points
            a0 = energies[0] * np.exp(-b0)
            popt, _ = curve_fit(exp_function, sf, energies, p0=[a0, b0], maxfev=1000)
            return popt[0]  # a + c at x=0
        except (RuntimeError, RuntimeWarning):
            print("Exponential fit failed, falling back to linear extrapolation")
            coeffs = np.polyfit(sf, energies, 1)
            return coeffs[1]  # intercept at 0
        
def calc_probability(prob, factor):
    return min(1.0, prob * factor)

# SCALE FACTORS FOR ZNE
scale_factors_20 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10.5]
scale_factors_10 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
scale_factors = [1.0, 2.0, 3.0]

estimators_zne = []

def prepare_estimators_zne_exact():
    estimators_zne.clear()
    for factor in scale_factors:
        scaled_error1 = depolarizing_error(calc_probability(depolarizing_prob1, factor), 1)
        scaled_error2 = depolarizing_error(calc_probability(depolarizing_prob2, factor), 2)

        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(scaled_error1, ['r'])
        scaled_noise.add_all_qubit_quantum_error(scaled_error2, ['cx'])

        #readout_prob_scaled = calc_probability(readout_prob, factor)
        #scaled_readout = ReadoutError([[1 - readout_prob_scaled, readout_prob_scaled], [readout_prob_scaled, 1 - readout_prob_scaled]])

        #scaled_noise.add_all_qubit_readout_error(scaled_readout)

        estimator = Estimator(
            backend_options={
                "noise_model": scaled_noise
            },
            run_options={
                "shots": None
            },
            approximation=True
        )

        estimators_zne.append(estimator)

def prepare_estimators_zne_5k():
    estimators_zne.clear()
    for factor in scale_factors:
        scaled_error1 = depolarizing_error(calc_probability(depolarizing_prob1, factor), 1)
        scaled_error2 = depolarizing_error(calc_probability(depolarizing_prob2, factor), 2)

        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(scaled_error1, ['r'])
        scaled_noise.add_all_qubit_quantum_error(scaled_error2, ['cx'])

        #readout_prob_scaled = calc_probability(readout_prob, factor)
        #scaled_readout = ReadoutError([[1 - readout_prob_scaled, readout_prob_scaled], [readout_prob_scaled, 1 - readout_prob_scaled]])

        #scaled_noise.add_all_qubit_readout_error(scaled_readout)

        estimator = Estimator(
            backend_options={
                "noise_model": scaled_noise
            },
            run_options={
                "shots": 5000
            }
        )
        estimators_zne.append(estimator)

def prepare_estimators_zne_100k():
    for factor in scale_factors:
        scaled_error1 = depolarizing_error(calc_probability(depolarizing_prob1, factor), 1)
        scaled_error2 = depolarizing_error(calc_probability(depolarizing_prob2, factor), 2)

        scaled_noise = NoiseModel()
        scaled_noise.add_all_qubit_quantum_error(scaled_error1, ['ry'])
        scaled_noise.add_all_qubit_quantum_error(scaled_error2, ['cx'])

        #readout_prob_scaled = calc_probability(readout_prob, factor)
        #scaled_readout = ReadoutError([[1 - readout_prob_scaled, readout_prob_scaled], [readout_prob_scaled, 1 - readout_prob_scaled]])

        #scaled_noise.add_all_qubit_readout_error(scaled_readout)

        estimator = Estimator(
            backend_options={
                "noise_model": scaled_noise
            },
            run_options={
                "shots": 100000
            }
        )
        estimators_zne.append(estimator)

def gate_noise_zne_richardson_exact(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='richardson')

def gate_noise_zne_linear_exact(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    if len(raw_energies) != len(scale_factors):
        print(f"Warning: {raw_energies} and {scale_factors} length mismatch")
        
    return extrapolate(scale_factors, raw_energies, method='linear')

def gate_noise_zne_exponential_exact(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='exponential')

def gate_noise_zne_richardson_5k(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='richardson')

def gate_noise_zne_linear_5k(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='linear')

def gate_noise_zne_exponential_5k(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='exponential')

def gate_noise_zne_richardson_100k(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='richardson')

def gate_noise_zne_linear_100k(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='linear')

def gate_noise_zne_exponential_100k(angles):
    raw_energies = []
    for est in estimators_zne:
        job = est.run([ansatz], [H], [angles])
        raw_energies.append(job.result().values[0])
    # extrapolate
    return extrapolate(scale_factors, raw_energies, method='exponential')

def execute_exact(circuit):
    job = estimator_gate_noise_exact.run([circuit], [H])
    return job.result().values[0]

def execute_5k(circuit):
    job = estimator_gate_noise_5k.run([circuit], [H])
    return job.result().values[0]

def execute_10k(circuit):
    job = estimator_gate_noise_10k.run([circuit], [H])
    return job.result().values[0]

def mitiq_extrapolate(angles, sf, method, shots):
    if method == 'linear':
        factory = zne.inference.LinearFactory(sf)
    elif method == 'richardson':
        factory = zne.inference.RichardsonFactory(sf)
    elif method == 'exponential':
        factory = zne.inference.ExpFactory(sf, asymptote=0.0)

    extrapolation_method = factory.extrapolate

    circuit = ansatz.assign_parameters(angles)
    folded_circuits = zne.construct_circuits(circuit=circuit, scale_factors=sf, scale_method=fold_gates_at_random)
    if shots == 5000:
        energies = [execute_5k(c) for c in folded_circuits]
    elif shots == 10000:
        energies = [execute_10k(c) for c in folded_circuits]
    elif shots == None:
        energies = [execute_exact(c) for c in folded_circuits]

    mitigated = zne.combine_results(sf, energies, extrapolation_method)
    return mitigated

def gate_noise_zne_mitiq_linear_exact(angles):
    return mitiq_extrapolate(angles, scale_factors, method='linear', shots=None)

def gate_noise_zne_mitiq_richardson_exact(angles):
    return mitiq_extrapolate(angles, scale_factors, method='richardson', shots=None)

def gate_noise_zne_mitiq_exponential_exact(angles):
    return mitiq_extrapolate(angles, scale_factors, method='exponential', shots=None)

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

def gate_noise_pec_mitiq_exact(angles, last_iteration):
    if last_iteration:
        mitigated = []
        for a in angles:
            # extrapolate
            circuit = ansatz.assign_parameters(a)
            reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level=depolarizing_prob2)
            mit = pec.execute_with_pec(circuit, execute_exact, representations=reps, num_samples=1000)
            mitigated.append(mit)
        return np.array(mitigated)
    else:
        return gate_noise_5k(angles)

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

def gate_noise_pec_mitiq_10k(angles, last_iteration):
    if last_iteration:
        mitigated = []
        for a in angles:
            # extrapolate
            circuit = ansatz.assign_parameters(a)
            reps = represent_operations_in_circuit_with_local_depolarizing_noise(circuit, noise_level=depolarizing_prob2)
            mit = pec.execute_with_pec(circuit, execute_10k, representations=reps, num_samples=1000)
            mitigated.append(mit)
        return np.array(mitigated)
    else:
        return gate_noise_10k(angles)

def noiseless(angles):
    job = estimator_noiseless.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def shot_noise_5k(angles):
    job = estimator_shot_noise_5k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def shot_noise_1k(angles):
    job = estimator_shot_noise_1k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def shot_noise_10k(angles):
    job = estimator_shot_noise_10k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def shot_noise_100k(angles):
    job = estimator_shot_noise_100k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def shot_noise_100(angles):
    job = estimator_shot_noise_100.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def gate_noise_exact(angles):
    job = estimator_gate_noise_exact.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def gate_noise_5k(angles):
    job = estimator_gate_noise_5k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def gate_noise_10k(angles):
    job = estimator_gate_noise_10k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def gate_noise_1k(angles):
    job = estimator_gate_noise_1k.run([ansatz], [H], [angles])
    energy = job.result().values[0]
    return energy

def fakeManilaV2_5k(angles):
    job = estimator_fakeManilaV2.run([transpiled_ansatz_manila], [H_5q], [angles])
    energy = job.result().values[0]
    return energy

def fakeAthensV2_5k(angles):
    job = estimator_fakeAthensV2.run([transpiled_ansatz_athens], [H_5q], [angles])
    energy = job.result().values[0]
    return energy

def fakeBogotaV2_5k(angles):
    job = estimator_fakeBogotaV2.run([transpiled_ansatz_bogota], [H_5q], [angles])
    energy = job.result().values[0]
    return energy

def fakeManilaV2_exact(angles):
    job = estimator_fakeManilaV2_exact.run([transpiled_ansatz_manila], [H_5q], [angles])
    energy = job.result().values[0]
    return energy

def fakeAthensV2_exact(angles):
    job = estimator_fakeAthensV2_exact.run([transpiled_ansatz_athens], [H_5q], [angles])
    energy = job.result().values[0]
    return energy

def fakeBogotaV2_exact(angles):
    job = estimator_fakeBogotaV2_exact.run([transpiled_ansatz_bogota], [H_5q], [angles])
    energy = job.result().values[0]
    return energy