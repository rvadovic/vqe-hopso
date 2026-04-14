from mpi4py import MPI
import numpy as np

#from costF.costF_2q_IvaH2_qiskit import cost_function_1 as cost_fn_h2
#from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import noiseless, shot_noise_1k, fakeManilaV2, fakeAthensV2, fakeBogotaV2, gate_noise_1k, shot_noise_5k, shot_noise_100, prepare_estimators_zne_100k, gate_noise_5k, gate_noise_zne_richardson_5k, prepare_estimators_zne_5k, shot_noise_5k, gate_noise_zne_mitiq_linear_5k, gate_noise_zne_mitiq_richardson_5k, gate_noise_zne_mitiq_exponential_5k, gate_noise_pec_mitiq_5k, gate_noise_zne_linear_5k, gate_noise_zne_exponential_5k, gate_noise_pec_mitiq_100k, gate_noise_zne_linear_100k, gate_noise_zne_mitiq_linear_100k, gate_noise_zne_mitiq_richardson_100k, gate_noise_zne_mitiq_exponential_100k, gate_noise_zne_richardson_100k, gate_noise_100k, shot_noise_100k
from src.costF.costF_4q_H2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import E_exact
#from costF.costF_8q_LiH import cost_fn_8qlih
#from costF.costF_8q_LiH import ansatz as ansatz_lih
from src.optimizers.hopso_final_mpi import mpi_hopso
#from src.optimizers.pso_mpi import pso_mpi
from src.utils.result_handler_csv import write_to_csv
from time import perf_counter

def set_seed(base_seed, run, rank):
    seed = base_seed + run * 10000 + rank
    return np.random.default_rng(seed)

# Define
optimizer = mpi_hopso
cost_F = shot_noise_100k
hp = [1, 1, 2*np.pi, 0.07]
num_particles = 12
runs = 100
dimension = ansatz_h2.num_parameters
maxcut = 2.05
max_iterations = 500

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Synchronize all processes
comm.Barrier()

# Print process information
print(f"Process {rank}/{size} ready")

# Broadcast initial parameters to all nodes
#total_tasks = runs * num_particles

if size != num_particles:
    if rank == 0:
        print(f"Warning: Number of cores ({size}) doesn't match total tasks ({num_particles})")
    
    # Adjust num_particles to match available cores
    num_particles = size

    if rank == 0:
        print(f"Adjusted to {num_particles} particles per run and {1} particles per rank")

# Create node-aware communicator
#node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)

if rank == 0:
    print(f"Initialization complete. Starting optimization with {runs} runs and {num_particles} particles per run and {1} particles per rank")
    results = []

if(cost_F.__name__ == "gate_noise_zne_richardson_100k" or cost_F.__name__ == "gate_noise_zne_linear_100k" or cost_F.__name__ == "gate_noise_zne_exponential_100k"):
        prepare_estimators_zne_100k()

if(cost_F.__name__ == "gate_noise_zne_richardson_5k" or cost_F.__name__ == "gate_noise_zne_linear_5k" or cost_F.__name__ == "gate_noise_zne_exponential_5k"):
        prepare_estimators_zne_5k()

# Run HOPSO
for i in range(runs):
    # Another barrier before starting main computation
    rng = set_seed(42, i, rank)
    comm.Barrier()
    if(rank == 0):
        start_time = perf_counter()

    e = mpi_hopso(cost_F, hp, dimension, maxcut, max_iterations, comm, rng)
    comm.Barrier()
    if(rank == 0):
        end_time = perf_counter()
        time = end_time - start_time
        results.append({"run": i+1, "final_energy": e, "time": time})

if(rank == 0):
    write_to_csv(cost_F.__name__, optimizer.__name__, results)
