import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid
from digits_data import load_data

X_train, Y_train = load_data()
print(f"X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}")

model = Sequential([
    tf.keras.Input(shape=(36,)),
    tf.keras.layers.Dense(25, activation='relu', name='Layer1'),
    tf.keras.layers.Dense(15, activation='relu', name='Layer2'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='Layer3')
])

model.summary()
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
)
model.fit(X_train, Y_train, epochs=100)
yhat = (model.predict(X_train) >= 0.5).astype(int)
print(yhat)
fig, ax = plt.subplots(2, 4, figsize=(8,8))
for i, ax in enumerate(ax.flat):
    print(i)
    X_reshape = X_train[i].reshape((6, 6))
    ax.imshow(X_reshape, cmap='grey')
    ax.set_title(str(Y_train[i])+', '+ str(yhat[i]))
    ax.set_axis_off()
plt.show()

X_test = np.array([
    [   0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0 ]
])

print((model.predict(X_test) >= 0.5).astype(int))
