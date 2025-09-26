import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets

import random

def load_data(w, b, nsamples, xstride=2, variance=20, features = 1):
    """ Load data.
        return x_data, y_data
    """
    x_data = np.zeros((nsamples, features))
    y_data = np.zeros((nsamples, 1))
    for i in range(nsamples):
        x_data[i, 0] = i*xstride
        for k in range(1, features):
            x_data[i, k] = i*xstride**(k*2)
        y_data[i] = np.log(0.5*(random.randint(0, variance) + w*x_data[i,0] + b))*(variance*np.cos(2*np.pi*x_data[i,0]))
        
    return x_data, y_data

features = 1
X, y = load_data(2, 0, 30, 4, 50, features)
x_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=1)
x_test, x_cv, y_test, y_cv = train_test_split(x_, y_, test_size=0.5, random_state=1)


models = []
polys = []
scalers = []
train_mses = []
cv_mses = []

ndegrees = 6

model_targets = []

x = np.arange(np.max(x_train),step=0.1).reshape(-1, 1)
for i in range(features - 1):
    x = np.hstack([x, x])
print(x)
    
for i in range(1, ndegrees):
    
    poly = PolynomialFeatures(degree=i, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)
    x_cv_mapped = poly.transform(x_cv)
    x_mapped = poly.transform(x)
    polys.append(poly) 
    
    standard_scaler = StandardScaler()
    x_train_mapped_scaled = standard_scaler.fit_transform(x_train_mapped)
    x_cv_mapped_scaled = standard_scaler.transform(x_cv_mapped)
    x_mapped_scaled = standard_scaler.transform(x_mapped)
    scalers.append(standard_scaler)

    model = LinearRegression()
    model.fit(x_train_mapped_scaled, y_train)
    models.append(model)

    yhat = model.predict(x_train_mapped_scaled)
    train_mse = mean_squared_error(yhat, y_train)/2
    train_mses.append(train_mse)

    yhat = model.predict(x_mapped_scaled)
    model_targets.append(yhat)

    yhat = model.predict(x_cv_mapped_scaled)
    cv_mse = mean_squared_error(yhat, y_cv)/2
    cv_mses.append(cv_mse)


degrees = range(1, ndegrees)
fig, ax = plt.subplots(1,2, figsize=(8, 8))
ax[0].scatter(x_train[:,0], y_train, c='r', marker='x', label='training')
ax[0].scatter(x_cv[:,0], y_cv, c='b', marker='o', label='cross validation')
ax[0].scatter(x_test[:,0], y_test, c='g', marker='v', label='test')
for i in range(len(model_targets)):
    ax[0].plot(x, model_targets[i], label='model'+str(i))
ax[1].plot(degrees, train_mses, c='r', label='training MSEs')
ax[1].plot(degrees, cv_mses, c='b', label='Cross Validation MSEs')
plt.legend()
plt.show()

degree = np.argmin(cv_mses)
print(degree)

degree = 6
lambdas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

l_train_mses = []
l_cv_mses = []

polys = []
scalers = []
model_targets = []
for i in range(len(lambdas)):
    poly = PolynomialFeatures(degree=degree+1, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)
    x_cv_mapped = poly.transform(x_cv)
    x_mapped = poly.transform(x)
    polys.append(poly) 
    
    standard_scaler = StandardScaler()
    x_train_mapped_scaled = standard_scaler.fit_transform(x_train_mapped)
    x_cv_mapped_scaled = standard_scaler.transform(x_cv_mapped)
    x_mapped_scaled = standard_scaler.transform(x_mapped)
    scalers.append(standard_scaler)

    model = Ridge(alpha=lambdas[i])
    model.fit(x_train_mapped_scaled, y_train)
    models.append(model)

    yhat = model.predict(x_train_mapped_scaled)
    l_train_mse = mean_squared_error(yhat, y_train)/2
    l_train_mses.append(l_train_mse)

    yhat = model.predict(x_mapped_scaled)
    model_targets.append(yhat)

    yhat = model.predict(x_cv_mapped_scaled)
    l_cv_mse = mean_squared_error(yhat, y_cv)/2
    l_cv_mses.append(l_cv_mse)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot: data + models
ax[0].scatter(x_train[:,0], y_train, c='r', marker='x', label='training')
ax[0].scatter(x_cv[:,0], y_cv, c='b', marker='o', label='cross validation')
ax[0].scatter(x_test[:,0], y_test, c='g', marker='v', label='test')

for i in range(len(model_targets)):
    ax[0].plot(x, model_targets[i], label=f'model {i}')

ax[0].set_title("Data & Models")
ax[0].legend()

# Right subplot: error curves
ax[1].plot(lambdas, l_train_mses, c='r', label='training MSEs')
ax[1].plot(lambdas, l_cv_mses, c='b', label='Cross Validation MSEs')

ax[1].set_title("Error vs Lambda")
ax[1].set_xlabel("Lambda")
ax[1].set_ylabel("MSE")
ax[1].legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)  # leave space for slider

# Left subplot: data + models
ax[0].scatter(x_train[:,0], y_train, c='r', marker='x', label='training')
ax[0].scatter(x_cv[:,0], y_cv, c='b', marker='o', label='cross validation')
ax[0].scatter(x_test[:,0], y_test, c='g', marker='v', label='test')

# Initial line with model[0]
(line,) = ax[0].plot(x, model_targets[0], label='model 0')

ax[0].set_title("Data & Model (choose with slider)")
ax[0].legend()

# Right subplot: error curves
ax[1].plot(lambdas, l_train_mses, c='r', label='training MSEs')
ax[1].plot(lambdas, l_cv_mses, c='b', label='Cross Validation MSEs')
ax[1].set_title("Error vs Lambda")
ax[1].set_xlabel("Lambda")
ax[1].set_ylabel("MSE")
ax[1].legend()

# --- Slider setup ---
ax_slider = plt.axes([0.25, 0.1, 0.55, 0.03])
slider = Slider(ax_slider, "Model index", 0, len(model_targets)-1, 
                valinit=0, valstep=1)

# --- Update function ---
def update(val):
    idx = int(slider.val)
    line.set_ydata(model_targets[idx])
    line.set_label(f'model {idx}')
    ax[0].legend()
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
