import numpy as np
data = np.genfromtxt('scaled_values_5k_3factors.csv', delimiter=',')
means = np.mean(data, axis=0)
print(means)