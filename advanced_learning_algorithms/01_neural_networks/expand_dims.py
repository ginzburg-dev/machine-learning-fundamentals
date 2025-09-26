import numpy as np

x = np.array([[1], [2]])

print(x)
x = np.hstack([x, np.onces()])

print(x)
