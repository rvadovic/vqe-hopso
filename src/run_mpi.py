from mpi4py import MPI
import numpy as np

#from costF.costF_2q_IvaH2_qiskit import cost_function_1 as cost_fn_h2
#from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import cost_function_noiseless, prepare_estimators_zne, cost_function_gate_noise, cost_function_gate_noise_zne, cost_function_shot_noise
from src.costF.costF_4q_H2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import E_exact
#from costF.costF_8q_LiH import cost_fn_8qlih
#from costF.costF_8q_LiH import ansatz as ansatz_lih
from src.optimizers.hopso_final_mpi import hopso
from src.utils.result_handler_csv import write_to_csv
from time import perf_counter

# Define 
optimizer = hopso
cost_F = cost_function_gate_noise
hp = [1, 1, 2*np.pi, 0.0583]
num_particles = 12 
particles_per_rank = 2
runs = 10
dimension = ansatz_h2.num_parameters
maxcut = 2.05
max_iterations = 500
e_min = []

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

if(cost_F.__name__ == "cost_function_gate_noise_zne"):
        prepare_estimators_zne()

# Run HOPSO
for i in range(runs):
    # Another barrier before starting main computation
    comm.Barrier()
    start_time = perf_counter()
    hopso(cost_F, hp, i, dimension, maxcut, e_min, particles_per_rank, max_iterations, comm)
    comm.Barrier()
    end_time = perf_counter()
    if(rank == 0):
        time = end_time - start_time
        results.append({"run": i+1, "final_energy": np.min(e_min), "time": time})

if(rank == 0):
    write_to_csv(cost_F.__name__, optimizer.__name__, results)
