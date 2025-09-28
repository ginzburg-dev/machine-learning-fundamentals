import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility

df = pd.read_csv('/Users/dmitryginzburg/Development/machine-learning-fundamentals/advanced_learning_algorithms/01_neural_networks/heart_dataset.csv')
print(df.head())

# split  cathegorical cariables with n>3 outputs into n binary variables. one-hot encoding
cat_variables = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)
print(df.head())

features = [x for x in df.columns if x not in 'HeartDisease'] # removing our target values
print(features)
print(len(features))

#help(train_test_split)

# split dataset into train and test sets
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=RANDOM_STATE)
print(f'train samples: {len(X_train)}')
print(f'validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')

# sklearn decision tree 
min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]

# test sampling list
accuracy_list_train = []
accuracy_list_val = []

for min_samples_split in min_samples_split_list:
    model = DecisionTreeClassifier(min_samples_split=min_samples_split,
                                random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.set_title('DecisionTree. Min samples split test. Train x Validation metrics')
ax.set_xlabel('min_samples_split')
ax.set_ylabel('accuracy')
ax.plot(min_samples_split_list, accuracy_list_train)
ax.plot(min_samples_split_list, accuracy_list_val)
ax.legend(['Train','Validation'])
plt.show()

# test depth list
accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth,
                                random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.set_title('DecisionTree. Max depth test. Train x Validation metrics')
ax.set_xlabel('max_depth')
ax.set_ylabel('accuracy')
ax.plot(max_depth_list, accuracy_list_train)
ax.plot(max_depth_list, accuracy_list_val)
ax.legend(['Train','Validation'])
plt.show()

# now we can choose the best values for our model:
max_depth = 3
min_samples_split = 50

decision_tree_model = DecisionTreeClassifier(min_samples_split = 50, max_depth = 3, random_state=RANDOM_STATE)
decision_tree_model.fit(X_train, y_train)

print(f'Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train), y_train):.4f}')
print(f'Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val), y_val):.4f}')

# random forest min split samples test
accuracy_list_train = []
accuracy_list_val = []

for min_samples_split in min_samples_split_list:
    model = RandomForestClassifier(min_samples_split=min_samples_split,
                                random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.set_title('Random Forest. Min samples split test. Train x Validation metrics')
ax.set_xlabel('min_samples_split')
ax.set_ylabel('accuracy')
ax.plot(min_samples_split_list, accuracy_list_train)
ax.plot(min_samples_split_list, accuracy_list_val)
ax.legend(['Train','Validation'])
plt.show()

# Random forest test depth list
accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth=max_depth,
                                random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.set_title('Random Forest. Max depth test. Train x Validation metrics')
ax.set_xlabel('max_depth')
ax.set_ylabel('accuracy')
ax.plot(max_depth_list, accuracy_list_train)
ax.plot(max_depth_list, accuracy_list_val)
ax.legend(['Train','Validation'])
plt.show()

# Random forest n_estimators test
n_estimators_list = [10, 50, 100, 500]
accuracy_list_train = []
accuracy_list_val = []

for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators,
                                random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

fig, ax = plt.subplots(1,1, figsize=(8, 8))
ax.set_title('Random Forest. N_Estimators test. Train x Validation metrics')
ax.set_xlabel('max_depth')
ax.set_ylabel('accuracy')
ax.plot(n_estimators_list, accuracy_list_train)
ax.plot(n_estimators_list, accuracy_list_val)
ax.legend(['Train','Validation'])
plt.show()


# the best parameters for Random Forest are
max_depth = 16
min_samples_split = 10
n_estimators = 100

randdom_forest_model = RandomForestClassifier(n_estimators=100,
                                            max_depth=16,
                                            min_samples_split=10, random_state=RANDOM_STATE).fit(X_train, y_train)
print(f'Metrics train:\n\tAccuracy score: {accuracy_score(randdom_forest_model.predict(X_train), y_train):.4f}')
print(f'Metrics validation:\n\tAccuracy score: {accuracy_score(randdom_forest_model.predict(X_val), y_val):.4f}')


# XGBoost 

n = int(len(X_train)*0.8)
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1, verbosity = 1, early_stopping_rounds=10, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit, y_train_fit, eval_set=[(X_train_eval, y_train_eval)])

print(xgb_model.best_iteration)
print(f'Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train), y_train):.4f}')
print(f'Metrics validation:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val), y_val):.4f}')
