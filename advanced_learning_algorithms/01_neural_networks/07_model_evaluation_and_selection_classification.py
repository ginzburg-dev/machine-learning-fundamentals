import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid

import random

# load data
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

X = X[y != 2] # only two classes
y = y[y != 2]
print(X.shape, y.shape)

# plot data
fig, ax = plt.subplots(1,1, figsize=(8, 8))
x_0 = X[y == 0]; x_1 = X[y == 1]
ax.scatter(x_0[:,0], x_0[:,1], c='r', label='data 1')
ax.scatter(x_1[:,0], x_1[:,1], c='b', label='data 2')
plt.legend()
plt.show()

x_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=50)
x_test, x_cv, y_test, y_cv = train_test_split(x_, y_, test_size=0.4, random_state=50)

print(f"the shape of the training set(input) is: {x_train.shape}")
print(f"the shape of the training set(target) is: {y_train.shape}")
print(f"the shape of the cross validation set(input) is: {x_cv.shape}")
print(f"the shape of the cross validation set(target) is: {y_cv.shape}")
print(f"the shape of the test set(input) is: {x_test.shape}")
print(f"the shape of the test set(target) is: {y_test.shape}")

standard_scaler = StandardScaler()

x_train_scaled = standard_scaler.fit_transform(x_train)
x_cv_scaled = standard_scaler.fit_transform(x_cv)
x_test_scaled = standard_scaler.fit_transform(x_test)

# build train models

nn_models = [
    Sequential([
        tf.keras.layers.Dense(25, activation='relu', name='L1'),
        tf.keras.layers.Dense(15, activation='relu', name='L2'),
        tf.keras.layers.Dense(1, activation='linear', name='L3')
    ]),
    Sequential([
        tf.keras.layers.Dense(20, activation='relu', name='L1'),
        tf.keras.layers.Dense(12, activation='relu', name='L2'),
        tf.keras.layers.Dense(12, activation='relu', name='L3'),
        tf.keras.layers.Dense(20, activation='relu', name='L4'),
        tf.keras.layers.Dense(1, activation='linear', name='L5')
    ]),
    Sequential([
        tf.keras.layers.Dense(32, activation='relu', name='L1'),
        tf.keras.layers.Dense(16, activation='relu', name='L2'),
        tf.keras.layers.Dense(8, activation='relu', name='L3'),
        tf.keras.layers.Dense(4, activation='relu', name='L4'),
        tf.keras.layers.Dense(12, activation='relu', name='L5'),
        tf.keras.layers.Dense(1, activation='linear', name='L6')
    ])
]

nn_train_error = []
nn_cv_error = []

for model in nn_models:
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    )

    print(f"Training {model.name}...")

    model.fit(
        x_train_scaled, y_train,
        epochs = 100,
        verbose = 0
    )

    print("Done!\n")

    threshold = 0.5
    yhat = model.predict(x_train_scaled)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= threshold, 1, 0)
    train_error = np.mean(yhat != y_train)
    nn_train_error.append(train_error)

    yhat = model.predict(x_cv_scaled)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= threshold, 1, 0)
    cv_error = np.mean(yhat != y_train)
    nn_cv_error.append(cv_error)

# Print the result
for model_num in range(len(nn_train_error)):
    print(
        f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
        f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
        )

model_num = 3
yhat = nn_models[model_num - 1].predict(x_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
nn_test_error = np.mean(yhat != y_test)
print(f"Selected Model: {model_num}")
print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
print(f"Test Set Classification Error: {nn_test_error:.4f}")
