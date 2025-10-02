import numpy as np
import matplotlib.pyplot as plt
import random

def make_blobs(centers, nsamples=200, std=0.1):
    n = len(centers) * nsamples
    x_train = np.zeros((n, 2), dtype=np.float32)
    y_train = np.zeros((n,), dtype=np.int32)
    for i in range(len(centers)):
        for j in range(nsamples):
            x_dist = ( random.randint(1, 10000) / 10000 ) * std
            y_dist = ( random.randint(1, 10000) / 10000 ) * std
            x_train[i*nsamples + j] = centers[i] + np.array([x_dist, y_dist])
            y_train[i*nsamples + j] = i
    return x_train, y_train

centers = np.array([[10, 20]])
X_train, Y_train = make_blobs(centers, nsamples=1000, std=100)

plt.hist(np.log(X_train[:,0]**4), bins=50)
plt.show()
