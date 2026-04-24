import numpy as np
from scipy.optimize import minimize

# Configuration
max_iter = 6001

def cobyla(cost_fn, dimension, budget, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=(dimension))
    result = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': budget, 'tol': 1e-20, 'rhobeg': 0.5})
    return result.fun, result.x