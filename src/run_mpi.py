from mpi4py import MPI
import numpy as np

#from costF.costF_2q_IvaH2_qiskit import cost_function_1 as cost_fn_h2
#from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import cost_function_noiseless, prepare_estimators_zne, cost_function_gate_noise, cost_function_gate_noise_zne_1510, cost_function_shot_noise
from src.costF.costF_4q_H2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import E_exact
#from costF.costF_8q_LiH import cost_fn_8qlih
#from costF.costF_8q_LiH import ansatz as ansatz_lih
from src.optimizers.hopso_final_mpi import mpi_hopso
from src.optimizers.async_hopso_mpi import mpi_ahopso
#from src.optimizers.pso_mpi import pso_mpi
from src.utils.result_handler_csv import write_to_csv
from time import perf_counter

# Define
optimizer = mpi_hopso
cost_F = cost_function_gate_noise_zne_1510
hp = [1, 1, 2*np.pi, 0.058333]
num_particles = 4 
particles_per_rank = 1
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

if size != num_particles/particles_per_rank:
    if rank == 0:
        print(f"Warning: Number of cores ({size}) doesn't match total tasks ({num_particles/particles_per_rank})")
    
    # Adjust num_particles to match available cores
    num_particles = size*particles_per_rank

    if rank == 0:
        print(f"Adjusted to {num_particles} particles per run and {particles_per_rank} particles per rank")

# Create node-aware communicator
#node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)

if rank == 0:
    print(f"Initialization complete. Starting optimization with {runs} runs and {num_particles} particles per run and {particles_per_rank} particles per rank")
    results = []

if(cost_F.__name__ == "cost_function_gate_noise_zne_1510"):
        prepare_estimators_zne()

# Run HOPSO
for i in range(runs):
    # Another barrier before starting main computation
    comm.Barrier()
    if(rank == 0):
        start_time = perf_counter()

    e = mpi_hopso(cost_F, hp, dimension, maxcut, particles_per_rank, max_iterations, comm)
    comm.Barrier()
    if(rank == 0):
        end_time = perf_counter()
        time = end_time - start_time
        results.append({"run": i+1, "final_energy": e, "time": time})

if(rank == 0):
    write_to_csv(cost_F.__name__, optimizer.__name__, results)
