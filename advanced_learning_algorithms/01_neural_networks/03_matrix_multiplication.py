import numpy as np 

A = np.array([
    [1, -1, 0.1],
    [2, -2, 0.2]
])

AT = A.T # transpose

W = np.array([
    [3, 5, 7, 9],
    [4, 6, 8, 0]
])

Z = np.matmul(AT, W)
Zalt = AT @ W # alternative way to mult matrices
print(Z)
print(Zalt)

print(W[:, 0])
