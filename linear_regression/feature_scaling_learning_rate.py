import numpy as np
import matplotlib.pyplot as plt

def f_wb_function(x, w, b):
    f_wb = np.dot(x,w) + b
    return f_wb

def cost_function(x, y, w, b):
    cost = 0
    m = x.shape[0]
    for i in range(m):
        f_wb = f_wb_function(x[i], w, b)
        cost += (f_wb - y[i])**2
    cost /= 2 * m
    return cost

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = f_wb_function(x[i],w,b) - y[i]
        for j in range(n):
            dj_dw[j] += err*x[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, iterations, alpha):
    J_hist = []
    w_final = np.zeros((x.shape[1]))
    b_final = 0.

    for i in range(iterations):
        dj_dw_i, dj_db_i = compute_gradient(x, y, w_final, b_final)
        w_final = w_final - alpha * dj_dw_i
        b_final = b_final - alpha * dj_db_i
        J_hist.append([cost_function(x,y,w_final,b_final), w_final.copy(), b_final])

        if i % 1 == 0:
            outstr = f"Iteration {i}: {J_hist[-1][0]:.2f}, "
            for j in range(len(dj_dw_i)):
                outstr += f"w{j}: {w_final[j]}, "
            #outstr += f"b: {b_final}, "
            #for j in range(len(dj_dw_i)):
            #    outstr += f"djdw{j}: {dj_dw_i[j]}, "
            print(outstr)
    print(f"w, b found by gradient descent: w:{w_final}, b: {b_final}")
    return w_final, b_final, J_hist
    
def mean_normalize(X):
    mean = np.mean(X,axis=0)
    max = np.max(X,axis=0)
    min = np.min(X,axis=0)
    normalized = (X - mean) / (max - min)
    return normalized, mean, max, min

def z_score_normalize(X):
    mean = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    normalized = (X - mean) / sigma
    return normalized, mean, sigma

def predict(data, w, b, mean, sigma):
    data_norm = (data - mean)/sigma
    #print(data_norm)
    data_predict = np.dot(data_norm, w) + b
    return data_predict


def plot_train_data(x, y, features):
    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x[:,i], y)
        ax[i].set_xlabel(features[i])
    ax[0].set_ylabel("Price (1000's)")
    plt.show()

def plot_cost_i_w(x, y, hist, step):
    fig, ax = plt.subplots(1, 2, figsize=(12,3))
    iter = np.arange(len(hist))
    costs = np.array([item[0] for item in hist])
    w0 = np.array([item[1][0] for item in hist])
    w0_x = np.arange(np.min(w0), np.max(w0),step)
    cost_y = np.zeros((w0_x.shape[0],))
    for i in range(w0_x.shape[0]):
        w_for_test = np.zeros_like(hist[0][1])
        w_for_test[0] = w0_x[i]
        cost_y[i] = cost_function(x, y, w_for_test, 0)
    ax[0].plot(iter,costs); ax[0].set_xlabel("Iterations"); ax[0].set_ylabel("Cost")
    ax[1].plot(w0,costs); ax[1].set_xlabel("w[0]"); ax[1].set_ylabel("Cost")
    ax[1].plot(w0_x,cost_y)
    plt.show()

def plot_scaled_features(X):
    X_mean,_,_,_ = mean_normalize(X)
    X_zscore,_,_ = z_score_normalize(X)

    fig, ax = plt.subplots(1,3, figsize=(12, 3))
    ax[0].scatter(X[:,0], X[:,3])
    ax[0].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[3])
    ax[0].set_title("unnormalized")
    ax[0].axis('equal')

    ax[1].scatter(X_mean[:,0], X_mean[:,3])
    ax[1].set_xlabel(X_features[0])
    ax[1].set_ylabel(X_features[3])
    ax[1].set_title(r"(X - $\mu$)/(min-max)")
    ax[1].axis('equal')

    ax[2].scatter(X_zscore[:,0], X_zscore[:,3])
    ax[2].set_xlabel(X_features[0])
    ax[2].set_ylabel(X_features[3])
    ax[2].set_title(r"Z-score normalized")
    ax[2].axis('equal')
    plt.tight_layout(rect=[0.05,0.05,0.95,0.95]) # packing the layout
    fig.suptitle("distribution of features (raw, mean normalized, z-score normalized)")
    plt.show()

def plot_normalized_data(x, y, features_name):
    X_norm,_,_ = z_score_normalize(x)

    fig,ax = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(len(ax)):
        ax[i].hist(X_norm[:,i])
        ax[i].set_xlabel(features_name[i])
    fig.suptitle("distribution of features after normalization")
    plt.tight_layout(rect=[0,0.05,1,0.95])
    plt.show()

X_train = np.array([[1.24e+03, 3.00e+00, 1.00e+00, 6.40e+01], 
        [1.95e+03, 3.00e+00, 2.00e+00, 1.70e+01],
        [1.72e+03, 3.00e+00, 2.00e+00, 4.20e+01],
        [1.96e+03, 3.00e+00, 2.00e+00, 1.50e+01],
        [1.31e+03, 2.00e+00, 1.00e+00, 1.40e+01],
        [8.64e+02, 2.00e+00, 1.00e+00, 6.60e+01],
        [1.84e+03, 3.00e+00, 1.00e+00, 1.70e+01],
        [1.03e+03, 3.00e+00, 1.00e+00, 4.30e+01],
        [3.19e+03, 4.00e+00, 2.00e+00, 8.70e+01],
        [7.88e+02, 2.00e+00, 1.00e+00, 8.00e+01],
        [1.20e+03, 2.00e+00, 2.00e+00, 1.70e+01],
        [1.56e+03, 2.00e+00, 1.00e+00, 1.80e+01],
        [1.43e+03, 3.00e+00, 1.00e+00, 2.00e+01],
        [1.22e+03, 2.00e+00, 1.00e+00, 1.50e+01],
        [1.09e+03, 2.00e+00, 1.00e+00, 6.40e+01],
        [8.48e+02, 1.00e+00, 1.00e+00, 1.70e+01],
        [1.68e+03, 3.00e+00, 2.00e+00, 2.30e+01],
        [1.77e+03, 3.00e+00, 2.00e+00, 1.80e+01],
        [1.04e+03, 3.00e+00, 1.00e+00, 4.40e+01],
        [1.65e+03, 2.00e+00, 1.00e+00, 2.10e+01],
        [1.09e+03, 2.00e+00, 1.00e+00, 3.50e+01],
        [1.32e+03, 3.00e+00, 1.00e+00, 1.40e+01],
        [1.59e+03, 0.00e+00, 1.00e+00, 2.00e+01],
        [9.72e+02, 2.00e+00, 1.00e+00, 7.30e+01],
        [1.10e+03, 3.00e+00, 1.00e+00, 3.70e+01],
        [1.00e+03, 2.00e+00, 1.00e+00, 5.10e+01],
        [9.04e+02, 3.00e+00, 1.00e+00, 5.50e+01],
        [1.69e+03, 3.00e+00, 1.00e+00, 1.30e+01],
        [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
        [1.42e+03, 3.00e+00, 2.00e+00, 1.90e+01],
        [1.16e+03, 3.00e+00, 1.00e+00, 5.20e+01],
        [1.94e+03, 3.00e+00, 2.00e+00, 1.20e+01],
        [1.22e+03, 2.00e+00, 2.00e+00, 7.40e+01],
        [2.48e+03, 4.00e+00, 2.00e+00, 1.60e+01],
        [1.20e+03, 2.00e+00, 1.00e+00, 1.80e+01],
        [1.84e+03, 3.00e+00, 2.00e+00, 2.00e+01],
        [1.85e+03, 3.00e+00, 2.00e+00, 5.70e+01],
        [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01],
        [1.10e+03, 2.00e+00, 2.00e+00, 9.70e+01],
        [1.78e+03, 3.00e+00, 2.00e+00, 2.80e+01],
        [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01],
        [1.78e+03, 4.00e+00, 2.00e+00, 1.07e+02],
        [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
        [1.55e+03, 3.00e+00, 1.00e+00, 1.60e+01],
        [1.95e+03, 3.00e+00, 2.00e+00, 1.60e+01],
        [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01],
        [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01],
        [8.16e+02, 2.00e+00, 1.00e+00, 5.80e+01],
        [1.35e+03, 3.00e+00, 1.00e+00, 2.10e+01],
        [1.57e+03, 3.00e+00, 1.00e+00, 1.40e+01],
        [1.49e+03, 3.00e+00, 1.00e+00, 5.70e+01],
        [1.51e+03, 2.00e+00, 1.00e+00, 1.60e+01],
        [1.10e+03, 3.00e+00, 1.00e+00, 2.70e+01],
        [1.76e+03, 3.00e+00, 2.00e+00, 2.40e+01],
        [1.21e+03, 2.00e+00, 1.00e+00, 1.40e+01],
        [1.47e+03, 3.00e+00, 2.00e+00, 2.40e+01],
        [1.77e+03, 3.00e+00, 2.00e+00, 8.40e+01],
        [1.65e+03, 3.00e+00, 1.00e+00, 1.90e+01],
        [1.03e+03, 3.00e+00, 1.00e+00, 6.00e+01],
        [1.12e+03, 2.00e+00, 2.00e+00, 1.60e+01],
        [1.15e+03, 3.00e+00, 1.00e+00, 6.20e+01],
        [8.16e+02, 2.00e+00, 1.00e+00, 3.90e+01],
        [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01],
        [1.39e+03, 3.00e+00, 1.00e+00, 6.40e+01],
        [1.60e+03, 3.00e+00, 2.00e+00, 2.90e+01],
        [1.22e+03, 3.00e+00, 1.00e+00, 6.30e+01],
        [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02],
        [2.60e+03, 4.00e+00, 2.00e+00, 2.20e+01],
        [1.43e+03, 3.00e+00, 1.00e+00, 5.90e+01],
        [2.09e+03, 3.00e+00, 2.00e+00, 2.60e+01],
        [1.79e+03, 4.00e+00, 2.00e+00, 4.90e+01],
        [1.48e+03, 3.00e+00, 2.00e+00, 1.60e+01],
        [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01],
        [1.43e+03, 3.00e+00, 1.00e+00, 2.20e+01],
        [1.16e+03, 3.00e+00, 1.00e+00, 5.30e+01],
        [1.55e+03, 3.00e+00, 2.00e+00, 1.20e+01],
        [1.98e+03, 3.00e+00, 2.00e+00, 2.20e+01],
        [1.06e+03, 3.00e+00, 1.00e+00, 5.30e+01],
        [1.18e+03, 2.00e+00, 1.00e+00, 9.90e+01],
        [1.36e+03, 2.00e+00, 1.00e+00, 1.70e+01],
        [9.60e+02, 3.00e+00, 1.00e+00, 5.10e+01],
        [1.46e+03, 3.00e+00, 2.00e+00, 1.60e+01],
        [1.45e+03, 3.00e+00, 2.00e+00, 2.50e+01],
        [1.21e+03, 2.00e+00, 1.00e+00, 1.50e+01],
        [1.55e+03, 3.00e+00, 2.00e+00, 1.60e+01],
        [8.82e+02, 3.00e+00, 1.00e+00, 4.90e+01],
        [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01],
        [1.04e+03, 3.00e+00, 1.00e+00, 6.20e+01],
        [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01],
        [8.03e+02, 2.00e+00, 1.00e+00, 8.00e+01],
        [1.43e+03, 3.00e+00, 2.00e+00, 2.10e+01],
        [1.66e+03, 3.00e+00, 1.00e+00, 6.10e+01],
        [1.54e+03, 3.00e+00, 1.00e+00, 1.60e+01],
        [9.48e+02, 3.00e+00, 1.00e+00, 5.30e+01],
        [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01],
        [1.43e+03, 2.00e+00, 1.00e+00, 4.30e+01],
        [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01],
        [1.21e+03, 3.00e+00, 1.00e+00, 2.00e+01],
        [1.05e+03, 2.00e+00, 1.00e+00, 6.50e+01]])

y_train = np.array([300., 509.8,  394.,   540.,   415.,  230.,   560.,   294.,   718.2,  200.,
    302.,   468.,   374.2,  388.,   282.,   311.8,  401.,   449.8,  301.,   502.,
    340.,   400.28, 572.,   264.,   304.,   298.,   219.8,  490.7,  216.96, 368.2,
    280.,   526.87, 237.,   562.43, 369.8,  460.,   374.,   390.,   158.,   426.,
    390.,   277.77, 216.96, 425.8,  504.,   329.,   464.,   220.,   358.,   478.,
    334.,   426.98, 290.,   463.,   390.8,  354.,   350.,   460.,   237.,   288.3,
    282.,   249.,   304.,   332.,   351.8,  310.,   216.96, 666.34, 330.,   480.,
    330.3,  348.,   304.,   384.,   316.,   430.4,  450.,   284.,   275.,   414.,
    258.,   378.,   350.,   412.,   373.,   225.,   390.,   267.4,  464.,   174.,
    340.,   430.,   440.,   216.,   329.,   388.,   390.,   356.,   257.8, ])

X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# run gradient descent
_, _, hist = gradient_descent(X_train, y_train, 10, alpha=1e-7)
plot_cost_i_w(X_train, y_train, hist, step=0.001)

plot_train_data(X_train, y_train, X_features)
plot_scaled_features(X_train)
plot_normalized_data(X_train, y_train, X_features)

# run gradient descent with normalized data
X_normalized,x_mean,x_sigma = z_score_normalize(X_train)
w_final_norm, b_final_norm, hist_norm = gradient_descent(X_normalized, y_train, 1000, alpha=1e-1)
plot_cost_i_w(X_normalized, y_train, hist_norm, step=0.1)

# plot targer vs prediction using z-score normalized model
m = X_normalized.shape[0]
y_predict = np.zeros(m)
for i in range(m):
    y_predict[i] = np.dot(X_normalized[i], w_final_norm) + b_final_norm
fig,ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i], y_predict, label='predict')
ax[0].set_ylabel("Price"); ax[0].legend()
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.suptitle("targer vs prediction using z-score normalized model")
plt.show()

# predict a house price using our trained model
x_house = np.array([1200, 3, 1, 40])
x_house_predict = predict(x_house,w_final_norm,b_final_norm, x_mean, x_sigma)
print(f"predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
