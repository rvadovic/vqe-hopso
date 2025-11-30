from qiskit.quantum_info import SparsePauliOp
import numpy as np

import qiskit_aer
print(qiskit_aer.__version__)
import qiskit
print("Qiskit version:", qiskit.__version__)

H = SparsePauliOp(['XXIIIIII', 'YYIIIIII', 'IIIIXXII', 'IIIIYYII', 'XZXIIIII', 'YZYIIIII', 'IIIIXZXI', 'IIIIYZYI', 'IXZXIIII', 'IYZYIIII', 'IIIIIXZX', 'IIIIIYZY', 'IIXXIIII', 'IIYYIIII', 'IIIIIIXX', 'IIIIIIYY', 'IIIIIIII', 'ZIIIIIII', 'IIIIZIII', 'ZIIIZIII', 'IIIIIIII', 'IZIIIIII', 'IIIIIZII', 'IZIIIZII', 'IIIIIIII', 'IIZIIIII', 'IIIIIIZI', 'IIZIIIZI', 'IIIIIIII', 'IIIZIIII', 'IIIIIIIZ', 'IIIZIIIZ'],
              coeffs=[-0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,
 -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,
 -0.5+0.j, -0.5+0.j,  0. +0.j, -0. +0.j, -0. +0.j,  0. +0.j,  0. +0.j,
 -0. +0.j, -0. +0.j,  0. +0.j,  0. +0.j, -0. +0.j, -0. +0.j,  0. +0.j,
  0. +0.j, -0. +0.j, -0. +0.j,  0. +0.j])

#Exact ground state energy of 4-spin subspace -4.00

from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister
# estimator = Estimator(approximation=True, run_options={"shots": None})
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
# from qiskit_aer.primitives import Estimator
# estimator = Estimator(approximation=True, run_options={"shots": 1000})

qc = QuantumCircuit(8)

qc.h(0)           
qc.h(2)           
qc.h(4)           
qc.h(6)           
qc.x(1)           
qc.x(3)           
qc.x(5)           
qc.x(7)           
qc.cx(0, 1)   
qc.cx(2, 3)    
qc.cx(4, 5)    
qc.cx(6, 7)    
qc.cx(1, 2)    
qc.cx(5, 6)    
qc.cry(np.pi/2, 2, 1)   
qc.cry(np.pi/2, 6, 5)   
qc.cx(1, 2)   
qc.cx(5, 6)    
qc.cz(2, 3)    
qc.cz(6, 7)    

qc.barrier()

def add_ansatz_layer(qc, theta):
    assert len(theta) == 20, "Each layer requires exactly 20 parameters."

    # Section 1: On-site CRZ (Coulomb interaction)
    qc.crz(theta[0], 4, 0)
    qc.crz(theta[1], 5, 1)
    qc.crz(theta[2], 6, 2)
    qc.crz(theta[3], 7, 3)
    qc.barrier()
    # Section 2: Sequential hopping
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        qc.cx(a, b)
    qc.crx(theta[4], 1, 0)
    qc.crx(theta[5], 3, 2)
    qc.crx(theta[6], 5, 4)
    qc.crx(theta[7], 7, 6)
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        qc.cx(a, b)
    qc.crz(theta[8], 0, 1)
    qc.crz(theta[9], 2, 3)
    qc.crz(theta[10], 4, 5)
    qc.crz(theta[11], 6, 7)
    qc.barrier()
    # Cross links
    for a, b in [(1, 2), (5, 6)]:
        qc.cx(a, b)
    qc.crx(theta[12], 2, 1)
    qc.crx(theta[13], 6, 5)
    for a, b in [(1, 2), (5, 6)]:
        qc.cx(a, b)
    qc.crz(theta[14], 1, 2)
    qc.crz(theta[15], 5, 6)
    qc.barrier()
    # Vertical links (non-sequential)
    for a, b in [(0, 3), (4, 7)]:
        qc.cx(a, b)
    qc.crx(theta[16], 3, 0)
    qc.crx(theta[17], 7, 4)
    for a, b in [(0, 3), (4, 7)]:
        qc.cx(a, b)
    qc.crz(theta[18], 0, 3)
    qc.crz(theta[19], 7, 4)
    qc.barrier()

num_layers = 2

theta = [Parameter(f'theta_{i}') for i in range(20 * num_layers)]

for i in range(num_layers):
    theta_block = theta[i * 20 : (i + 1) * 20]
    add_ansatz_layer(qc, theta_block)  # from previous definition

def apply_hopping(qc, q1, q2):
    qc.cx(q1, q2)
    qc.ch(q2,q1)
    qc.cx(q1, q2)

apply_hopping(qc, 0, 1)
apply_hopping(qc, 2, 3)
apply_hopping(qc, 4, 5)
apply_hopping(qc, 6, 7)
qc.barrier()
apply_hopping(qc, 1,2)
apply_hopping(qc, 5,6)
qc.barrier()
apply_hopping(qc, 0,3)
apply_hopping(qc, 4,7)

ansatz = qc

def objective_function_1(angle):
    bound_circuit = ansatz.assign_parameters(angle)
   
    # Create the input for the StatevectorEstimator
    estimator_input = [(bound_circuit, H)]
   
    # Run the estimator
    job = estimator.run(estimator_input)
    result = job.result()
   
    # Based on the inspection output, the correct way to access the value is:
    # result[0].data.evs
    pub_result = result[0]
    data_bin = pub_result.data
    energy = float(np.real(data_bin.evs))
   
    return energy