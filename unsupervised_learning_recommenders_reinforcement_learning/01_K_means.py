import numpy as np
import matplotlib.pyplot as plt

import random

def make_blobs(centers, nsamples=200, std=0.1):
    n = len(centers) * nsamples
    x_train = np.zeros((n, 2), dtype=np.float32)
    y_train = np.zeros((n,), dtype=np.int32)
    for i in range(len(centers)):
        for j in range(nsamples):
            x_dist = ( random.randint(1, 1000) / 1000 ) * std
            y_dist = ( random.randint(1, 1000) / 1000 ) * std
            x_train[i*nsamples + j] = centers[i] + np.array([x_dist, y_dist])
            y_train[i*nsamples + j] = i
    return x_train, y_train

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i, x in enumerate(X):
        min_centroid = np.linalg.norm(x - centroids[0])
        idx_i = 0
        for k in range(len(centroids)):
            dist = np.linalg.norm(x - centroids[k])
            if dist < min_centroid:
                min_centroid = dist
                idx_i = k
        idx[i] = idx_i
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                        example in X. Concretely, idx[i] contains the index of 
                        the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = [X[i] for i in range(len(idx)) if idx[i] == k]
        centroids[k] = sum(points)/len(points)
    return centroids

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def run_kMean(X, initial_centroids, max_iter=10):
    """Run K-Mean algorithm on data matrix X, where each row of X is a single example"""
    m, n = X.shape
    centroids =initial_centroids
    for i in range(max_iter):
        print(f'K-Means iteration {i}/{max_iter}')
        idx =  find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, len(centroids))
    return centroids, idx

centers = np.array([[10, 20], [50, 60], [100, 120]])
X_train, Y_train = make_blobs(centers, nsamples=200, std=20)

K = 3
max_iter = 100
#initial_centroids = np.array([[20, 30], [30, 30], [70, 60]])
initial_centroids = kMeans_init_centroids(X_train, K)
idx_init = find_closest_centroids(X_train, initial_centroids)

centroids, idx = run_kMean(X_train, initial_centroids, max_iter)

fig, ax = plt.subplots(1,2)
red_center_init = X_train[idx_init == 0]
green_center_init = X_train[idx_init == 1]
blue_center_init = X_train[idx_init == 2]
red_center = X_train[idx == 0]
green_center = X_train[idx == 1]
blue_center = X_train[idx == 2]
ax[0].scatter(red_center_init[:,0], red_center_init[:,1], marker='o', c='r', label='K=1')
ax[0].scatter(green_center_init[:,0], green_center_init[:,1], marker='v', c='g', label='K=2')
ax[0].scatter(blue_center_init[:,0], blue_center_init[:,1], marker='x', c='b', label='K=3')
ax[0].scatter(initial_centroids[:,0], initial_centroids[:,1], marker='x', c='purple', label='centroids')
ax[0].set_title('Init state')
ax[0].legend()
ax[1].scatter(red_center[:,0], red_center[:,1], marker='o', c='r', label='K=1')
ax[1].scatter(green_center[:,0], green_center[:,1], marker='v', c='g', label='K=2')
ax[1].scatter(blue_center[:,0], blue_center[:,1], marker='x', c='b', label='K=3')
ax[1].scatter(centroids[:,0], centroids[:,1], marker='x', c='purple', label='centroids')
ax[1].set_title('K-mean')
ax[1].legend()
plt.show()

# load image
original_img = plt.imread('unsupervised_learning_recommenders_reinforcement_learning/bird_small.png')
plt.imshow(original_img)
plt.show()

print(f'Shape of original image is: {original_img.shape}')

X_img = np.reshape(original_img, (original_img.shape[0]*original_img.shape[1], 3))
print(X_img.shape)

K = 16
max_iter = 20
initial_centroids = kMeans_init_centroids(X_img, K)
idx_init = find_closest_centroids(X_img, initial_centroids)
centroids, idx = run_kMean(X_img, initial_centroids, max_iter)
X_img_recovered = centroids[idx, :]
X_img_recovered = np.reshape(X_img_recovered, original_img.shape)

print(f'Shape of idx: {idx.shape}')
print(f'Closest point for the first five elements: {idx[:5]}')

# Display original image
fig, ax = plt.subplots(1,3)
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_img_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()

# Display color palette
centroids = np.reshape(centroids, (centroids.shape[0], 1, 3))
ax[2].imshow(centroids)
ax[2].set_title('Color pallette')
ax[2].set_axis_off()
plt.show()
