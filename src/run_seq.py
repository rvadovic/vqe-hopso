import numpy as np
#from costF.costF_2q_IvaH2_qiskit import objective_function_1 as cost_fn_h2
#from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
#from costF.costF_8q_LiH import cost_fn_8qlih
#from costF.costF_8q_LiH import ansatz as ansatz_8qlih
from src.costF.costF_4q_H2_qiskit import (
    noiseless,
    shot_noise_5k,
    gate_noise_5k,
    gate_noise_exact,
    gate_noise_zne_linear_5k,
    gate_noise_zne_linear_exact,
    prepare_estimators_zne_exact,
    prepare_estimators_zne_5k,
    gate_noise_zne_richardson_exact
)
from src.costF.costF_4q_H2_qiskit import ansatz as ansatz_h2
from src.costF.costF_4q_H2_qiskit import E_exact
from src.optimizers.hopso_final import hopso
from src.optimizers.cobyla import cobyla
from src.optimizers.de import de
from src.optimizers.pso import pso
from time import perf_counter
from src.utils.result_handler_csv import write_to_csv


optimizers = [cobyla,de]
costfs = [gate_noise_zne_richardson_exact]


runs = 100
dimension = ansatz_h2.num_parameters





for optimizer in optimizers:
    for costf in costfs:
        budget = 6000
        if(costf.__name__.startswith("gate_noise_zne")):
            if(costf.__name__.endswith("exact")):
                prepare_estimators_zne_exact()
            elif(costf.__name__.endswith("5k")):
                prepare_estimators_zne_5k()
            budget = 2000
        results = []
        print(f"Running {optimizer.__name__} on {costf.__name__}, budget: {budget}")
        for i in range(runs):
            seed = 42 + i*10000
            start_time = perf_counter()
            e, best_position = optimizer(costf, dimension, budget, seed)
            #hopso(cost_function_noiseless, [1,1,2*np.pi,0.05], 12, 10, ansatz_h2.num_parameters, 2.05, e_min)
            end_time = perf_counter()
            time = end_time - start_time
            results.append({"run": i+1, "final_energy": e, "time": time, "best_position": best_position.tolist()})
            
        write_to_csv(costf.__name__, optimizer.__name__, results)
    