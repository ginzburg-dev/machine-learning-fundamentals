import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid
import random

def make_blobs(centers, nsamples=200, std=0.1):
    n = len(centers) * nsamples
    x_train = np.zeros((n, 2), dtype=np.float32)
    y_train = np.zeros((n,), dtype=np.int32)
    for i in range(len(centers)):
        for j in range(nsamples):
            x_dist = ( random.randint(1, 1000) / 1000 ) * std
            y_dist = ( random.randint(1, 1000) / 1000 ) * std
            x_train[i*nsamples + j] = centers[i] + np.array([x_dist, y_dist])
            y_train[i*nsamples + j] = i
    return x_train, y_train

centers = np.array([[10, 20], [50, 60], [90, 80], [200, 150]])
X_train, Y_train = make_blobs(centers, nsamples=200, std=20)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train[:,0], X_train[:,1])
plt.show()

model = Sequential([
    tf.keras.layers.Dense(25, activation='relu', name='L1'),
    tf.keras.layers.Dense(15, activation='relu', name='L2'),
    tf.keras.layers.Dense(4, activation='linear', name='L3')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(0.003)
)

model.fit(X_train, Y_train, epochs=200)

pred = model.predict(X_train)
softmax = tf.nn.softmax(pred).numpy()

for i in range(4):
    print(np.argmax(softmax[i*200]))

fig, ax = plt.subplots(1,1)
ax.scatter(X_train[:,0], X_train[:,1])
plt.show()
