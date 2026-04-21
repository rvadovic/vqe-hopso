import numpy as np
data = np.genfromtxt('results\exponential_energies.csv', delimiter=',')
means = np.mean(data, axis=0)
print(means)