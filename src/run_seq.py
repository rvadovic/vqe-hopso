import numpy as np
#from costF.costF_2q_IvaH2_qiskit import objective_function_1 as cost_fn_h2
#from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
#from costF.costF_8q_LiH import cost_fn_8qlih
#from costF.costF_8q_LiH import ansatz as ansatz_8qlih
from costF.costF_4q_H2_qiskit import cost_function_noiseless
from costF.costF_4q_H2_qiskit import cost_function_shot_noise
from costF.costF_4q_H2_qiskit import cost_function_gate_noise
from costF.costF_4q_H2_qiskit import ansatz as ansatz_h2
from costF.costF_4q_H2_qiskit import E_exact    
from optimizers.hopso_final import hopso
from time import perf_counter

e_min = []

start_time = perf_counter()
hopso(cost_function_noiseless, [1,1,2*np.pi,0.05], 12, 10, ansatz_h2.num_parameters, 2.05, e_min)
end_time = perf_counter()
e_min = np.min(e_min)
error = abs(E_exact-e_min)
satisfies = error < 1.59e-3  # Chemical accuracy threshold
print(". e_min: " + str(e_min) + ", error: " + str(error) + ", satisfies: " + str(satisfies))
print(". time: " + str(end_time - start_time))