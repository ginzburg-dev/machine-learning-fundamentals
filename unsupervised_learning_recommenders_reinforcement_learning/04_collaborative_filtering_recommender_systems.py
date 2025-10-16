import numpy as np
import tensorflow as tf
from tensorflow import keras

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def normalize(Y, R):
    Y_mean = np.sum(Y*R, axis=1)/(np.sum(R, axis=1) + 1e-12).reshape(1, -1)
    Y_norm = Y - np.multiply(Y_mean, R)
    return Y_norm, Y_mean

R = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [5, 5, 0],
    [1, 2, 5],
    [5, 1, 1]
])

Y_norm, Y_mean = normalize(Y, R)
print(Y_norm, Y_mean)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float32),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float32),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float32),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 500
lambda_ = 1

for iter in range(iterations):
  with tf.GradientTape() as tape:
      loss = cofi_cost_func_v(X, W, b, Y_norm, R, lambda_)

  grad = tape.gradient(loss, [W, X, b])

  optimizer.apply_gradients(zip(grad, [W, X, b]))

  if iter % 20 == 0:
      print(f"Training lossa at iteration {iter}: {loss:.2f}")

p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
p = p + Y_mean

print(p)

# To check how close two films are to each other by their features, 
# use | x_1_k - x_1_j |^2 the squared distance.
