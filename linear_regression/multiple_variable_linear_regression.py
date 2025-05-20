import math, copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider

def prediction_single_loop(x, w, b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p += x[i]*w[i]
    p = p + b
    return p

def predict(x, w, b):
    p = np.dot(x,w) + b
    return p

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost /= 2 * m
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err*X[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iterations):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iterations):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        J_history.append(cost_function(X,y,w,b))

        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")
    return w, b, J_history

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
Y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


print(f"X Shape: {X_train.shape}, X Type:{type(X_train)}")
print(X_train)
print(f"Y Shape: {Y_train.shape}, Y Type:{type(Y_train)}")
print(Y_train)
print(f"W_init: {w_init}, B_init: {b_init}")

# get a row from our traininig data
x_vec = X_train[0,:]
print(f"x_vec: {x_vec}")

# make a loop prediction
f_wb_loop = prediction_single_loop(x_vec, w_init, b_init)
print(f"loop prediction: {f_wb_loop}")

# make a dot prediction
f_wb = predict(x_vec, w_init, b_init)
print(f"dot prediction: {f_wb}")

# compute cost using pre-cosen optimal parameters
cost = compute_cost(X_train, Y_train, w_init, b_init)
print(f"Cost at optimal w: {cost}")

# compute gradient
tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, Y_train, w_init, b_init)
print(f"dj_dw: {tmp_dj_dw}, dj_db: {tmp_dj_db}")

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(X_train, Y_train, initial_w, initial_b, 
                                            compute_cost, compute_gradient, 
                                            alpha, iterations)
print(f"w, b found by gradient descent: {w_final}, {b_final:0.2f}")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {Y_train[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
ax1.plot(J_hist)
ax2.plot( np.arange(len(J_hist[0:10:1])), J_hist[0:10:1])
ax1.set_title("Cost vs. iteration"); ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step'); ax2.set_xlabel('iteration step')
plt.show()
