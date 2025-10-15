import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """input: p - proportion of positive examples"""
    if p == 1 or p == 0:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

assert entropy(0.5) == 1
assert entropy(0) == 0
assert entropy(1) == 0

def weighted_entropy(X, y, left_indices, right_indices):
    """This function takes the splitted dataset, 
    the indices we chose to split and returns the weighted entropy"""
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

def information_gain(X, y, left_indices, right_indices):
    """X, y are elements in the node"""
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy

def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes,
        the left node has animals that feature = 1, the right node those that have the feature = 0.
        index feature 0 = ear shape (pointy | otherwise),
        index feature 1 = face shape (round | otherwise),
        index feature 2 = whoskers (present | absent)"""
    left_indices = []
    right_indices = []

    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

# X_train: for each example, contains 3 features:
# 
#       - Ear Shape (1 if pointy, 0 otherwise)
#       - Face Shape (1 if round, 0 otherwise)
#       - Whiskers (1 if present, 0 otherwise)
# y_train: whether the animal is a cat
# 
#       - 1 if the animal is a cat
#       - 0 otherwise

X_train = np.array([[1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

left_indices, right_indices = split_indices(X_train, 0)
we = weighted_entropy(X_train, y_train, left_indices, right_indices)
igain = information_gain(X_train, y_train, left_indices, right_indices)
print(f"weighted entropy: {we}")
print(f"information gain: {igain}")
