import sys, os
from gradient_descent_logistic import run_gradient_descent_logistic, sigmoid
import numpy as np 
import matplotlib.pyplot as plt

train_x = np.array([[0, 2] ,[1, 1], [2, 3], [3, 2], [4, 1], [5, 3]])
train_y = np.array([0, 0, 1, 1, 1, 1])

w0 = np.zeros_like(train_x[0])

w_res, b_res, j_hist = run_gradient_descent_logistic(train_x, train_y, w0=w0, b0=0.0, iterations=10000, alpha=1)

def plot_data():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    for i in range(train_y.shape[0]):
        if train_y[i] == 0:
            ax[0].scatter(train_x[i,0], train_x[i,1], marker='o', c='b', label = "real data")
        else:
            ax[0].scatter(train_x[i,0], train_x[i,1], marker='x', c='r', label = "real data")
    ax[0].plot([0, -b_res/w_res[0]], [-b_res/w_res[1], 0], label = "decision boundary")
    plt.show()

plot_data()
