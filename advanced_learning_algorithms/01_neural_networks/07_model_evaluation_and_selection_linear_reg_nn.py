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

def load_data(w, b, nsamples, xstride=2, variance=20):
    """ Load data.
        return x_data, y_data
    """
    x_data = np.zeros((nsamples, 1))
    y_data = np.zeros((nsamples, 1))
    for i in range(nsamples):
        x_data[i] = i*xstride
        y_data[i] = np.log(0.5*(random.randint(0, variance) + w*x_data[i] + b))
    return x_data, y_data

X_data, y_data = load_data(1, 0, 300, 2, 500)
print(X_data.shape, y_data.shape)

# split datasets into -> training, cross validation, test sets
x_train, x_, y_train, y_ = train_test_split(X_data, y_data, test_size=0.4, random_state=50)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=50)
del x_, y_

# plot data
fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.scatter(x_train, y_train, c='r', marker='x', label='training')
ax.scatter(x_cv, y_cv, c='b', marker='o', label='cross validation')
ax.scatter(x_test, y_test, c='g', marker='v', label='test')
plt.legend()
plt.show()

# feature scaling
scaler_linear = StandardScaler()
x_train_scaled = scaler_linear.fit_transform(x_train)
print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze(): 0.2f}")
print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze(): 0.2f}")

# train model
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)

yhat = linear_model.predict(x_train_scaled)

print(f"training MSE(sklearn): {mean_squared_error(y_train, yhat)/2}")

mean_squared_error_custom = 0.

for i in range(len(yhat)):
    mean_squared_error_custom += (y_train[i] - yhat[i])**2
mean_squared_error_custom /= 2*len(yhat)
print(f"training MSE(custom): {mean_squared_error_custom}")

# scale the cross validation set using the mean and standart deviation of the training set
x_cv_scaled = scaler_linear.transform(x_cv)
yhat = linear_model.predict(x_cv_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat)/2:0.2f}")

# add features
poly = PolynomialFeatures(degree = 2, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)

print(f"First 5 mapped x_train with new polynoimal features [x, x^2]: {x_train_mapped[:5]}")

# scale polynomial features
scaler_poly = StandardScaler()
x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
print(x_train_mapped_scaled[:5])

# train model with new polynomial features
model = LinearRegression()
model.fit(x_train_mapped_scaled, y_train)
yhat = model.predict(x_train_mapped_scaled)
print(f"Training MSE: {mean_squared_error(y_train, yhat)/2}")

x_cv_mapped = poly.transform(x_cv) # add poly features to cv
x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped) # scale poly features to cv
yhat = model.predict(x_cv_mapped_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat)/2}")

# loop over 10 degrees

models = []
scalers = []
polys = []
train_mses = []
cv_mses = []
n = 20
for i in range(1, n):
    poly = PolynomialFeatures(degree = i, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)

    scaler_poly = StandardScaler()
    x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
    scalers.append(scaler_poly)

    model = LinearRegression()
    model.fit(x_train_mapped_scaled, y_train)
    models.append(model)

    yhat = model.predict(x_train_mapped_scaled)
    train_mse = mean_squared_error(yhat, y_train)/2
    train_mses.append(train_mse)

    x_cv_mapped = poly.transform(x_cv)
    x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)

    yhat = model.predict(x_cv_mapped_scaled)
    cv_mse = mean_squared_error(yhat, y_cv)/2
    cv_mses.append(cv_mse)

# plot data
degrees = range(1, n)
fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.plot(degrees, train_mses, c='r', label='training MSEs')
ax.plot(degrees, cv_mses, c='b', label='cross validation MSEs')
ax.set_title('degree of polynomial vs. train and CV MSEs')
plt.legend()
plt.show()

# choosing the best model
best_degree =np.argmin(cv_mses)
print(f"Lowest CV MSE is found in the model with degree={best_degree + 1}")

x_test_mapped = polys[best_degree].transform(x_test)
x_test_mapped_scaled = scalers[best_degree].transform(x_test_mapped)

yhat = models[best_degree].predict(x_test_mapped_scaled)
test_mse = mean_squared_error(yhat, y_test)/2

print(f"Training MSE: {train_mses[best_degree]:0.2f}")
print(f"Cross Validation MSE: {cv_mses[best_degree]:0.2f}")
print(f"Test MSE: {test_mse:0.2f}")

# NN model selection ------------------

degree = 1
poly = PolynomialFeatures(degree=degree, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
x_cv_mapped = poly.transform(x_cv)
x_test_mapped = poly.transform(x_test)

scaler = StandardScaler()
x_train_mapped_scaled = scaler.fit_transform(x_train_mapped)
x_cv_mapped_scaled = scaler.transform(x_cv_mapped)
x_test_mapped_scaled = scaler.transform(x_test_mapped)

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

nn_train_mses = []
nn_cv_mses = []

for model in nn_models:

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
    )
    print(f"Training {model.name}...")

    # train model
    model.fit(
        x_train_mapped_scaled, y_train, epochs = 300,
        verbose = 0
    )

    print("Done!\n")

    # record training mse
    yhat = model.predict(x_train_mapped_scaled)
    train_mse = mean_squared_error(yhat, y_train) / 2
    nn_train_mses.append(train_mse)

    # record cv mse
    yhat = model.predict(x_cv_mapped_scaled)
    cv_mse = mean_squared_error(yhat, y_cv) / 2
    nn_cv_mses.append(cv_mse)

print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(  f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
            f"CV MSE: {nn_cv_mses[model_num]:.2f}")


