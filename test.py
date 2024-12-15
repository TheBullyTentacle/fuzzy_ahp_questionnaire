import numpy as np

a = np.identity(5)

for i in range(5):
    a[:, i] /= np.sum(a[:, i])
print(a)