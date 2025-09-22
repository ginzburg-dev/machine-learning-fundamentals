import numpy as np
import copy
import time

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def cost_logistic(X, y, w, b):
    """ Computes cost
        Args:
            X (ndarray (m,n)): Data, m examples with n features
            y (ndarray (m,)) : target values
            w (ndarray (n,)) : model parameters
            b (scalar)       : model parameter
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += - y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost /= m
    return cost

def calculate_gradient_logistic(x, y, w, b):
    """
    Computes the gradient for logistic regression 

    Args:
        X (ndarray (m,n): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters  
        b (scalar)      : model parameter
    Returns
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i*x[i,j]
        dj_db = dj_db + err_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
    

def run_gradient_descent_logistic(X, y, w0, b0, displStep=1000, iterations=10000, alpha=1e-3):
    m = X.shape[0]
    n = X.shape[1]
    w = copy.deepcopy(w0)
    b = b0
    j_hist = []

    start = time.time()

    for iter in range(iterations):
        dj_dw, dj_db = calculate_gradient_logistic(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        #if iter < 100000: 
        j_hist.append(cost_logistic(X, y, w, b))

        if iter % displStep == 0:
            print(f"Iteration {iter:4d}: Cost {j_hist[-1]}" )

    end = time.time()
    print(f"Extimated time: {(end - start):.4f} s")
    print(f"w, b found by gradient descent: w{w}, b{b}")
    return w, b, j_hist
        