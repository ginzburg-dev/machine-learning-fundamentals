import numpy as np
import matplotlib.pyplot as plt

def min_max_normalize(X):
    X_min = np.min(X)
    X_max = np.max(X)
    normalized = (X - X_min) / (X_max - X_min)
    return normalized

def mean_normalize(X):
    X_mean = np.mean(X)
    X_min = np.min(X)
    X_max = np.max(X)
    normalized = (X - X_mean) / (X_max - X_min)
    return normalized

def z_score_normalize(X):
    X_mean = np.mean(X)
    X_standard_deviation = np.std(X)
    normalized = (X - X_mean) / X_standard_deviation
    return normalized

X1_train = np.array([100, 293, 55, 303, 600, 600])
X2_train = np.array([0, 0.2, 100, 5, 4, 6])


fig, axs = plt.subplots(2, 2, figsize=(12,12))
(ax, ax0), (ax1, ax2) = axs # Unpacking

# Plot a real training data
ax.scatter(X1_train,X2_train)
ax.set_title("Real training data")
ax.set_xlabel('X1 feature')
ax.set_ylabel('X2 feature')

# Plot a min max normalized data
X1_min_max_normalized = min_max_normalize(X1_train)
X2_min_max_normalized = min_max_normalize(X2_train)

ax0.scatter(X1_min_max_normalized,X2_min_max_normalized)
ax0.set_title("Min max normalization")
ax0.set_xlabel('X1 scaled')
ax0.set_ylabel('X2 scaled')

# Plot a mean normalized data
X1_mean_normalized = mean_normalize(X1_train)
X2_mean_normalized = mean_normalize(X2_train)

ax1.scatter(X1_mean_normalized,X2_mean_normalized)
ax1.set_title("Mean normalization")
ax1.set_xlabel('X1 scaled')
ax1.set_ylabel('X2 scaled')

# Plot a z-score normalized data
X1_z_score_normalized = z_score_normalize(X1_train)
X2_z_score_normalized = z_score_normalize(X2_train)

ax2.scatter(X1_z_score_normalized,X2_z_score_normalized)
ax2.set_title("Z-score normalization")
ax2.set_xlabel('X1 scaled')
ax2.set_ylabel('X2 scaled')

plt.show()
