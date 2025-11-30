from mpi4py import MPI
import numpy as np
from costF.costF_2q_IvaH2_qiskit import objective_function_1 as cost_fn_h2
from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
from costF.costF_8q_LiH import cost_fn_8qlih
from costF.costF_8q_LiH import ansatz as ansatz_lih
from optimizers.hopso_final_mpi import hopso
from time import perf_counter

# Define parameters
hp = [1, 1, 2*np.pi, 0.05]
num_particles = 10
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

if size != num_particles:
    if rank == 0:
        print(f"Warning: Number of cores ({size}) doesn't match total tasks ({num_particles})")
    
    # Adjust num_particles to match available cores
    num_particles = size

    if rank == 0:
        print(f"Adjusted to {num_particles} particles per run")

# Create node-aware communicator
#node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)

if rank == 0:
    print(f"Initialization complete. Starting optimization with {runs} runs and {num_particles} particles per run")

# Run HOPSO
for i in range(runs):
    # Another barrier before starting main computation
    comm.Barrier()
    start_time = perf_counter()
    hopso(cost_fn_h2, hp, i, dimension, maxcut, e_min, max_iterations, comm)
    comm.Barrier()
    end_time = perf_counter()
    if(rank == 0): 
        print(str(i) + ". e_min: " + str(e_min[i]))
        print(str(i) + ". time: " + str(end_time - start_time))
