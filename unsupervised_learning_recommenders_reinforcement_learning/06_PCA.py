# PCA (Principal Component Analysys)
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


X = np.array([  [ 99,  -1],
                [ 98,  -1],
                [ 97,  -2],
                [101,   1],
                [102,   1],
                [103,   2]])

plt.plot(X[:,0], X[:,1], 'ro')
plt.show()

# Loading the PCA algorithm
pca_2 = PCA(n_components=2)
print(pca_2) # PCA(n_components=2)

# Let's fit the data. We do not need to scale it, since sklearn's implementation already handles it.
pca_2.fit(X)
print(pca_2.explained_variance_ratio_) # [0.99244289 0.00755711]

X_trans_2 = pca_2.transform(X)
print(X_trans_2)
# [[-1.38340578 -0.2935787 ]
# [-2.22189802  0.25133484]
# [-3.6053038  -0.04224385]
# [ 1.38340578  0.2935787 ]
# [ 2.22189802 -0.25133484]
# [ 3.6053038   0.04224385]]

pca_1 = PCA(n_components=1)
pca_1.fit(X)
print(pca_1.explained_variance_ratio_) # [0.99244289]

X_trans_1 = pca_1.transform(X)
print(X_trans_1)
# [[-1.38340578]
# [-2.22189802]
# [-3.6053038 ]
# [ 1.38340578]
# [ 2.22189802]
# [ 3.6053038 ]]

X_reduced_2 = pca_2.inverse_transform(X_trans_2)
X_reduced_1 = pca_1.inverse_transform(X_trans_1)
print(X_reduced_2)
# [ 98.  -1.]
# [ 97.  -2.]
# [101.   1.]
# [102.   1.]
# [103.   2.]]

# Transforming back from 1D to 2D causes a reduction of data, as it involves arranging it along a 2D plane.
print(X_reduced_1)
# [[ 98.84002499  -0.75383654]
# [ 98.13695576  -1.21074232]
# [ 96.97698075  -1.96457886]
# [101.15997501   0.75383654]
# [101.86304424   1.21074232]
# [103.02301925   1.96457886]]

