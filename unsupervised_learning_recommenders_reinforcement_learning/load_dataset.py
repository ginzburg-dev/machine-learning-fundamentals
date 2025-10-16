import numpy as np
import pandas as pd
from numpy import loadtxt

df = pd.read_csv('unsupervised_learning_recommenders_reinforcement_learning/film_raitings_small_dataset/ratings.csv')
X = df[['userId', 'movieId', 'rating', 'timestamp']].to_numpy(dtype=float)

print(X)
