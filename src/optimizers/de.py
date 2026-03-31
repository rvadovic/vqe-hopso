from scipy.optimize import differential_evolution
import numpy as np

# Configuration
popsize = 12
max_iter = 500

def de(cost_fn, dimension):
    bounds = [(-np.pi, np.pi)] * dimension
    result = differential_evolution(cost_fn, bounds, maxiter= max_iter, popsize=popsize, tol=1e-20)
    return result.fun

#dimension = 8    # note: 8*2*315 (dim*popsize*maxiter) = 5040 budget
