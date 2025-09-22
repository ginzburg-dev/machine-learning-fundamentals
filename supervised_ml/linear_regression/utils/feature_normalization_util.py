import numpy as np

def z_score_normalize(X):
    mean = np.mean(X, axis=0)
    standard_deviation = np.std(X, axis=0)
    normalized_data = (X - mean) / standard_deviation
    return normalized_data, mean, standard_deviation
