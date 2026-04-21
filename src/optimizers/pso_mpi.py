import numpy as np
from mpi4py import MPI


def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def mpi_pso(cost_fn, hp, dimension, max_iterations=500, comm=None, rng=None):
    # Unpack hyperparameters
    chi, c1, c2 = hp

    # Initialize position and velocity
    position = rng.uniform(-np.pi, np.pi, size = (dimension))
    particle_vel = rng.uniform(-np.pi, np.pi, size = (dimension))

    # Personal best
    personal_best_position = position.copy()
    personal_best_value = cost_fn(position)

    # Define rank
    rank = comm.Get_rank()

    # Initialize global best for the run
    all_personal_best_values = np.array(comm.allgather(personal_best_value))
    all_personal_best_positions = np.stack(comm.allgather(personal_best_position))

    # Global best
    global_best_idx = np.argmin(all_personal_best_values)
    global_best_value = all_personal_best_values[global_best_idx]
    global_best_position = all_personal_best_positions[global_best_idx] #.copy()


    iteration = 0
    while iteration < max_iterations:  
        r1, r2 = rng.random(dimension), rng.random(dimension)
        particle_vel = chi * (particle_vel + c1 * r1 * (personal_best_position - position) + c2 * r2 * (global_best_position - position))
        position = (position + particle_vel)

        current_value = cost_fn(position)

        if current_value < personal_best_value: 
            personal_best_value = current_value       
            personal_best_position = wrap_pi(position) # ensure pbest stays in [-π, π]

        all_personal_best_values = np.array(comm.allgather(personal_best_value))
        all_personal_best_positions = np.stack(comm.allgather(personal_best_position))

        current_best_idx = np.argmin(all_personal_best_values)
        current_best_val = all_personal_best_values[current_best_idx]
        
        if current_best_val < global_best_value:
            global_best_value = current_best_val
            global_best_position = all_personal_best_positions[current_best_idx].copy()
        
        iteration += 1
    
    if(rank == 0):
        return global_best_value, global_best_position