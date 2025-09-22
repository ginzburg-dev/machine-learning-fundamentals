import numpy as np
import time
from utils.feature_normalization_util import z_score_normalize

def f_wb_function(x, w, b):
    f_wb = np.dot(x,w) + b
    return f_wb

def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = f_wb_function(x[i], w, b)
        cost += (f_wb - y[i])**2
    cost /= 2 * m
    return cost

def calculate_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = f_wb_function(x[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += err*x[i, j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def run_gradient_descent(x, y, iterations, alpha, threshold=1e-6, print_step=100000):
    J_hist = []
    n = x.shape[1]
    w_result = np.zeros((n,))
    b_result = 0.
    start = time.time()
    for i in range(iterations):
        dj_dw_i, dj_db_i = calculate_gradient(x, y, w_result, b_result)
        w_result = w_result - alpha * dj_dw_i
        b_result = b_result - alpha * dj_db_i
        cost = cost_function(x, y, w_result, b_result)
        if len(J_hist) and abs(J_hist[-1][0] - cost) < threshold: 
            break
        J_hist.append([cost.copy(), w_result.copy(), b_result])

        if i % print_step == 0:
            outmsg = f"Iteration {i}: {J_hist[-1][0]}"
            #for j in range(n):
            #    outmsg += f"w{j}: {w_result[j]}, "
            print(outmsg)
        
    end = time.time()

    print(f"Extimated time: {(end - start):.4f} s")
    print(f"w, b found by gradient descent: w{w_result}, b{b_result}")
    return w_result, b_result, J_hist

def predict(data, w, b, mean, sigma) -> float:
    data_norm = (data - mean) / sigma
    predict = np.dot(data_norm, w) + b
    return predict
