import numpy as np
import matplotlib.pyplot as plt

def mean(X):
    n = X.shape[0]
    mean = 0
    for i in range(n):
        mean += X[i]
    mean /= n
    return mean

# Variance measures how much the values in a dataset spread out from the mean. Total squared error
def variance(X):
    n = X.shape[0]
    mean = np.mean(X)
    variance = 0
    for i in range(n):
        variance += (X[i] - mean)**2
    variance /= n
    return variance

# A measure of how spread out the data is around the mean. On average, how far is a value from the mean. Average error in original units
def standard_deviation(X):
    standard_deviation = np.sqrt(variance(X))
    return standard_deviation

def standard_deviation_custom(X):
    n = X.shape[0]
    standard_deviation = 0
    mean = np.mean(X)
    for i in range(n):
        standard_deviation += (X[i] - mean)**2
    standard_deviation /= n
    standard_deviation = np.sqrt(standard_deviation)
    return standard_deviation

X_train = np.array([100, 293, 55, 303, 600, 600])

print(f"N: {X_train.shape[0]}")

print(f"Custom mean: {mean(X_train)}")
print(f"Numpy mean: {np.mean(X_train)}")

print(f"Variance: {variance(X_train)}")
print(f"Numpy variance: {np.var(X_train)}")

print(f"Standart deviation: {standard_deviation(X_train)}")
print(f"Standart deviation custom: {standard_deviation_custom(X_train)}")
print(f"Numpy standart deviation: {np.std(X_train)}")
