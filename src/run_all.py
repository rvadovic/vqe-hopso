import subprocess
import itertools

# Define all configurations
cost_functions_old = ['shot_noise_10k', 'gate_noise_5k', 'gate_noise_exact', 'gate_noise_1k', 'gate_noise_zne_linear_5k', 'gate_noise_zne_linear_exact', 'gate_noise_zne_richardson_5k', 'gate_noise_zne_richardson_exact', 'gate_noise_zne_mitiq_linear_5k', 'gate_noise_zne_mitiq_richardson_5k', 'gate_noise_zne_mitiq_exponential_5k', 'gate_noise_zne_mitiq_linear_exact', 'gate_noise_zne_mitiq_richardson_exact', 'gate_noise_zne_mitiq_exponential_exact']
cost_functions = ['noiseless', 'shot_noise_5k', 'gate_noise_exact', 'gate_noise_5k', 'gate_noise_zne_linear_exact', 'gate_noise_linear_5k']
mitigation_methods = ['gate_noise_zne_linear_5k', 'gate_noise_zne_linear_exact', 'gate_noise_zne_richardson_5k', 'gate_noise_zne_exponential_exact', 'gate_noise_zne_exponential_5k', 'gate_noise_zne_richardson_exact', 'gate_noise_zne_mitiq_linear_5k', 'gate_noise_zne_mitiq_richardson_5k', 'gate_noise_zne_mitiq_exponential_5k', 'gate_noise_zne_mitiq_linear_exact', 'gate_noise_zne_mitiq_richardson_exact', 'gate_noise_zne_mitiq_exponential_exact']

for cost_fn in cost_functions:
    rank = 12
    budget = 6000
    if mitigation_methods.__contains__(cost_fn):
        rank = 4
        budget = 2000
    cmd = [
        "mpiexec", "-n", str(rank),
        "python", "-m", "src.run_mpi",
        "--cost_F", cost_fn,
        "--particles", str(rank),
        "--budget", str(budget)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)