import numpy as np
from scipy.optimize import minimize

# Configuration
max_iter = 6001

def cobyla(cost_fn, dimension):
    x0 = np.random.uniform(-np.pi, np.pi, size=(dimension))
    result = minimize(cost_fn, x0, method='COBYLA', options={'maxfun': max_iter, 'tol': 1e-20})
    return result.fun