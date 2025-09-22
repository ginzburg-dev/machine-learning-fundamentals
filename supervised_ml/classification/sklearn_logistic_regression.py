import numpy as np 
from sklearn.linear_model import LogisticRegression

train_x = np.array([[0, 2] ,[1, 1], [2, 3], [3, 2], [4, 1], [5, 3]])
train_y = np.array([0, 0, 1, 1, 1, 1])

lr_model = LogisticRegression()
lr_model.fit(train_x, train_y)

print(f"Predict X: {lr_model.predict(train_x)}")

print(f"Accuracy on train set: {lr_model.score(train_x, train_y)}")
