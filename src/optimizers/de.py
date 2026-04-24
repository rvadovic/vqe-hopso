from scipy.optimize import differential_evolution
import numpy as np

# Configuration
popsize = 80

def de(cost_fn, dimension, budget, seed):
    #popsize = budget // max_iter
    max_iter = budget // popsize
    rng = np.random.default_rng(seed)
    init_pop = rng.uniform(-np.pi, np.pi, size=(popsize, dimension))
    bounds = [(-np.pi, np.pi)] * dimension
    result = differential_evolution(cost_fn, bounds=bounds, maxiter= max_iter, init=init_pop, mutation=0.5, recombination=0.9, tol=1e-20, seed=seed, polish=False)
    return result.fun, result.x

#dimension = 8    # note: 8*2*315 (dim*popsize*maxiter) = 5040 budget
