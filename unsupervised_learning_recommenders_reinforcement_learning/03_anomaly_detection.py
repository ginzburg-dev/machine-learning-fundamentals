import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

def make_cloud_data(centers, nsamples=200, radius=2, power=10, threshold=10):
    # Sample random angles uniformly
    theta = np.random.uniform(0, 2*np.pi, nsamples)
    
    # Sample radius with higher density near 0
    # sqrt makes more samples near the center (try ^(1/2), ^(1/3), etc.)
    y_data = np.zeros((nsamples,))
    r = radius * np.sqrt(np.random.rand(nsamples)) ** power
    for i in range(len(r)):
        if r[i] > threshold:
            y_data[i] = 1
        else:
            y_data[i] = 0
    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.vstack((x, y)).T, y_data

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    mu = np.sum(X, axis=0)/m
    var = np.sum((X - mu)**2, axis=0)/m
    return mu, var

def gaussian(X, mu, var):
    p = (1/(2*np.pi*np.sqrt(var)))*np.exp((-(X - mu)**2)/(2*var))
    p = np.prod(p, axis=1)
    return p

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 10000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        tp = sum((predictions == 1) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        fn = sum((predictions == 0) & (y_val == 1))

        if tp + fp == 0 or tp + fn == 0:
            continue

        prec = tp/(tp + fp)
        recall = tp/(tp + fn)
        F1 = 2 * prec * recall/(prec + recall)

        

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

centers = np.array([[10, 20]])
X_train, y_train =  make_cloud_data(centers, 300, 100, 30, 50)
X_anom = X_train[y_train == 1]
X_norm = X_train[y_train == 0]


mu, var = estimate_gaussian(X_train)
p = gaussian(X_train, mu, var)

epsilon, f1 = select_threshold(y_train, p)  # returns best_epsilon, best_F1
outliers = p < epsilon

print("epsilon:", epsilon, "F1:", f1)

plt.scatter(X_norm[:,0], X_norm[:,1], marker='x', c='b')
plt.scatter(X_anom[:,0], X_anom[:,1], marker='x', c='r')
plt.scatter(X_train[outliers, 0], X_train[outliers, 1], marker='o', c='g')
plt.xlabel('latency')
plt.ylabel('throughoutput')
plt.show()
