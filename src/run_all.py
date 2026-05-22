import subprocess
import itertools

# Define all configurations
cost_functions_old = ['shot_noise_10k', 'gate_noise_5k', 'gate_noise_exact', 'gate_noise_1k', 'gate_noise_zne_linear_5k', 'gate_noise_zne_linear_exact', 'gate_noise_zne_richardson_5k', 'gate_noise_zne_richardson_exact', 'gate_noise_zne_mitiq_linear_5k', 'gate_noise_zne_mitiq_richardson_5k', 'gate_noise_zne_mitiq_exponential_5k', 'gate_noise_zne_mitiq_linear_exact', 'gate_noise_zne_mitiq_richardson_exact', 'gate_noise_zne_mitiq_exponential_exact']
cost_functions_backends = ['fakeAthensV2_5k_linear', 'fakeAthensV2_exact_linear']
backup = ['fakeBogotaV2_5k_linear', 'fakeBogotaV2_exact_linear', 'fakeManilaV2_5k_linear', 'fakeManilaV2_exact_linear']
cost_functions = [
    'gate_noise_zne_exponential_exact_def',
    'gate_noise_zne_exponential_5k_def',
    'gate_noise_zne_exponential_exact_real',
    'gate_noise_zne_exponential_5k_real',
    ]
mitigation_methods = ['gate_noise_zne_linear_5k', 'gate_noise_zne_linear_exact', 'gate_noise_zne_richardson_5k', 'gate_noise_zne_exponential_exact', 'gate_noise_zne_exponential_5k', 'gate_noise_zne_richardson_exact', 'gate_noise_zne_mitiq_linear_5k', 'gate_noise_zne_mitiq_richardson_5k', 'gate_noise_zne_mitiq_exponential_5k', 'gate_noise_zne_mitiq_linear_exact', 'gate_noise_zne_mitiq_richardson_exact', 'gate_noise_zne_mitiq_exponential_exact',
    'gate_noise_zne_linear_exact_def',
    'gate_noise_zne_linear_5k_def',
    'gate_noise_zne_linear_exact_real',
    'gate_noise_zne_linear_5k_real',
    'gate_noise_zne_richardson_exact_def',
    'gate_noise_zne_richardson_5k_def',
    'gate_noise_zne_richardson_exact_real',
    'gate_noise_zne_richardson_5k_real',
    'gate_noise_zne_exponential_exact_def',
    'gate_noise_zne_exponential_5k_def',
    'gate_noise_zne_exponential_exact_real',
    'gate_noise_zne_exponential_5k_real',
    ]
backend_mitigations = ['fakeAthensV2_5k_linear', 'fakeAthensV2_exact_linear', 'fakeBogotaV2_5k_linear', 'fakeBogotaV2_exact_linear', 'fakeManilaV2_5k_linear', 'fakeManilaV2_exact_linear']

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