from mpi4py import MPI
import numpy as np

#from costF.costF_2q_IvaH2_qiskit import cost_function_1 as cost_fn_h2
#from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import noiseless, prepare_estimators_zne_exact, shot_noise_1k, shot_noise_10k, fakeManilaV2, fakeAthensV2, fakeBogotaV2, gate_noise_1k, shot_noise_5k, shot_noise_100, prepare_estimators_zne_100k, gate_noise_5k, gate_noise_zne_richardson_5k, prepare_estimators_zne_5k, shot_noise_5k, gate_noise_zne_mitiq_linear_5k, gate_noise_zne_mitiq_richardson_5k, gate_noise_zne_mitiq_exponential_5k, gate_noise_pec_mitiq_5k, gate_noise_zne_linear_5k, gate_noise_zne_exponential_5k, gate_noise_pec_mitiq_10k, gate_noise_zne_linear_100k, gate_noise_zne_mitiq_linear_100k, gate_noise_zne_mitiq_richardson_100k, gate_noise_zne_mitiq_exponential_100k, gate_noise_zne_richardson_100k, gate_noise_10k, shot_noise_100k, gate_noise_exact, gate_noise_zne_linear_exact, gate_noise_zne_richardson_exact, gate_noise_zne_exponential_exact, gate_noise_zne_mitiq_exponential_exact, gate_noise_zne_mitiq_linear_exact, gate_noise_zne_mitiq_richardson_exact
from src.costF.costF_4q_H2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import E_exact
#from costF.costF_8q_LiH import cost_fn_8qlih
#from costF.costF_8q_LiH import ansatz as ansatz_lih
from src.optimizers.hopso_final_mpi import mpi_hopso
from src.optimizers.pso_mpi import mpi_pso
#from src.optimizers.pso_mpi import pso_mpi
from src.utils.result_handler_csv import write_to_csv
from time import perf_counter
import argparse

def set_seed(base_seed, run, rank):
    seed = base_seed + run * 10000 + rank
    return np.random.default_rng(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_F', type=str, required=True, help='Cost function to optimize')
    parser.add_argument('--particles', type=int, default=12, help='Number of particles')
    parser.add_argument('--budget', type=int, default=6000, help='Optimization budget')
    args = parser.parse_args()
    cost_F_mapping = {
        "noiseless": noiseless,
        "shot_noise_100": shot_noise_100,
        "shot_noise_5k": shot_noise_5k,
        "shot_noise_1k": shot_noise_1k,
        "shot_noise_10k": shot_noise_10k,
        "shot_noise_100k": shot_noise_100k,
        "fakeManilaV2": fakeManilaV2,
        "fakeAthensV2": fakeAthensV2,
        "fakeBogotaV2": fakeBogotaV2,
        "gate_noise_1k": gate_noise_1k,
        "gate_noise_5k": gate_noise_5k,
        "gate_noise_10k": gate_noise_10k,
        "gate_noise_exact": gate_noise_exact,
        "gate_noise_zne_linear_exact": gate_noise_zne_linear_exact,
        "gate_noise_zne_richardson_exact": gate_noise_zne_richardson_exact,
        "gate_noise_zne_exponential_exact": gate_noise_zne_exponential_exact,
        "gate_noise_zne_richardson_5k": gate_noise_zne_richardson_5k,
        "gate_noise_zne_linear_5k": gate_noise_zne_linear_5k,
        "gate_noise_zne_exponential_5k": gate_noise_zne_exponential_5k,
        "gate_noise_zne_mitiq_linear_exact": gate_noise_zne_mitiq_linear_exact,
        "gate_noise_zne_mitiq_richardson_exact": gate_noise_zne_mitiq_richardson_exact,
        "gate_noise_zne_mitiq_exponential_exact": gate_noise_zne_mitiq_exponential_exact,
        "gate_noise_zne_mitiq_linear_5k": gate_noise_zne_mitiq_linear_5k,
        "gate_noise_zne_mitiq_richardson_5k": gate_noise_zne_mitiq_richardson_5k,
        "gate_noise_zne_mitiq_exponential_5k": gate_noise_zne_mitiq_exponential_5k,
        "gate_noise_pec_mitiq_5k": gate_noise_pec_mitiq_5k,
        "gate_noise_zne_linear_100k": gate_noise_zne_linear_100k,
        "gate_noise_zne_mitiq_linear_100k": gate_noise_zne_mitiq_linear_100k,
        "gate_noise_zne_mitiq_richardson_100k": gate_noise_zne_mitiq_richardson_100k,
        "gate_noise_zne_mitiq_exponential_100k": gate_noise_zne_mitiq_exponential_100k,
        "gate_noise_zne_richardson_100k": gate_noise_zne_richardson_100k
    }

    cost_F = cost_F_mapping.get(args.cost_F)
    num_particles = args.particles
    # Define
    SCALING_FACTOR = 35
    budget = args.budget
    
    optimizer = mpi_pso
    runs = 100
    dimension = ansatz_h2.num_parameters
    maxcut = 2.05
    max_iterations = budget/num_particles
    hp_hopso = [1, 1, 2*np.pi, SCALING_FACTOR/max_iterations]
    hp_pso = [0.7298, 2.05, 2.05]


    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Synchronize all processes
    comm.Barrier()

    # Print process information
    #print(f"Process {rank}/{size} ready")

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
        print(f"PARTICLES: {num_particles}, BUDGET: {budget}, ITERATIONS: {max_iterations}, RUNS: {runs}, COST FUNCTION: {cost_F.__name__}")
        results = []

    if(cost_F.__name__.startswith("gate_noise_zne")):
        if(cost_F.__name__.endswith("exact")):
            prepare_estimators_zne_exact()
        elif(cost_F.__name__.endswith("5k")):
            prepare_estimators_zne_5k()
        elif(cost_F.__name__.endswith("100k")):
            prepare_estimators_zne_100k()

    # Run HOPSO
    for i in range(runs):
        # Another barrier before starting main computation
        rng = set_seed(42, i, rank)
        comm.Barrier()
        if(rank == 0):
            start_time = perf_counter()

        #e, best_position = mpi_hopso(cost_F, hp_hopso, dimension, maxcut, max_iterations, comm, rng)
        e, best_position = mpi_pso(cost_F, hp_pso, dimension, max_iterations, rng)
        comm.Barrier()
        if(rank == 0):
            end_time = perf_counter()
            time = end_time - start_time
            results.append({"run": i+1, "final_energy": e, "time": time, "best_position": best_position.tolist()})

    if(rank == 0):
        write_to_csv(cost_F.__name__, optimizer.__name__, results)

if __name__ == "__main__":
    main()