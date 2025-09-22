import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid

X_train = np.array([[1.0], [2.0]], dtype = np.float32)
Y_train = np.array([[300.0], [500.0]], dtype = np.float32)

#fig, ax = plt.subplots(1, 1)
#ax.scatter(X_train, Y_train, marker='x', c='r', label='Data Points')
#ax.legend( fontsize='xx-large')
#ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
#ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
#plt.show()

linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear')
print(linear_layer.get_weights())
a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)

w, b = linear_layer.get_weights()
print(f"w: {w}, b: {b}")

set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())


X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
print(X_train)

model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)

model.summary()

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)

set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
