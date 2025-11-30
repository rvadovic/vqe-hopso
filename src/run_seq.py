import numpy as np
from costF.costF_2q_IvaH2_qiskit import objective_function_1 as cost_fn_h2
from costF.costF_2q_IvaH2_qiskit import ansatz as ansatz_h2
from costF.costF_8q_LiH import cost_fn_8qlih
from costF.costF_8q_LiH import ansatz as ansatz_8qlih
from optimizers.hopso_final import hopso
from time import perf_counter

e_min = []

hopso(cost_fn_8qlih, [1,1,2*np.pi,0.05], 10, 10, ansatz_8qlih.num_parameters, 2.05, e_min)
print(e_min)