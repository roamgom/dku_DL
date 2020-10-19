#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import optuna
# import optuna.integration.lightgbm as oplgb
# import optuna.integration.xgboost as opxgb


# In[2]:


# get_ipython().system('ls ./data')


# In[3]:


# Const variable

DATA_PATH = 'data'
SEED = 222



train_data = pd.read_csv(f'{DATA_PATH}/trainset.csv',
                         header=None)
test_data = pd.read_csv(f'{DATA_PATH}/testset.csv',
                        header=None)

train_data[0]


# In[5]:


# Train Data 처리
# TODO:
# 1. data 전처리 [x]
#   label -> int 처리
# 1. Train dataset 나누기 (train/validation)
# 2. KMeans, KNN, DecisionTree, RandomForest
# 3. KFold
# 4. HyperParameter Tuning


# In[6]:


# 데이터 전처리
# train_data[0], uniques = pd.factorize(train_data[0])
# classify_values = {}

# for i, v in enumerate(uniques):
#     classify_values[i] = v
# classify_values
# train_data[1]


# In[7]:


X = train_data.iloc[:, 1:]
y = train_data.iloc[:, 0]
train_data
X


# In[8]:


# X
y


# In[9]:


# Define model

# KNN
knn = KNeighborsClassifier()
# DecisionTree
dt = DecisionTreeClassifier(random_state=SEED)
# SVM
svm = svm.SVC(random_state=SEED)
# RandomForest
rf = RandomForestClassifier(random_state=SEED,)

# LightGBM, Xgboost 는 optuna로 진행

models = [knn, dt, svm, rf]


# In[22]:


# Train model w. KFold

# KFold
kf = KFold(n_splits=5, random_state=SEED, shuffle=True)

mean_scores = []

# TODO: apply hyperparameter tuning
knn_params = {
    ''
}
dt_params = {
    
}
svm_params = {
    
}
rf_params = {
    
}


for model in models:
    print(f"{model} training\n")
    scores = cross_val_score(model, X, y, n_jobs=-1, scoring='accuracy', cv=kf)
    print(f"mean score: {np.mean(scores)}\n")
    mean_scores.append(np.mean(scores))
print(mean_scores)


# In[10]:


# LightGBM, Xgboost w. optuna

# LightGBM
lightgbm = lgb.LGBMClassifier(random_state=SEED)
# Xgboost
xgbboost = xgb.XGBClassifier(random_state=SEED)

def objective(trial):
    
    classifier = trial.suggest_categorical('classifier', ['KNeighbor', 'DecisionTree', 'SVM', 
                                                          'RandomForest', 'LightGBM', 'Xgboost'])
    
    # KFold
    kf = KFold(n_splits=8, random_state=SEED, shuffle=True)
    
    if classifier == 'KNeighbor':
        # KNN params
        knn_n_neighbors = trial.suggest_int('n_neighbors', 3, 10)
        knn_weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        knn_algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        knn_leaf_size = trial.suggest_int('leaf_size', 10, 40)
        knn_p = trial.suggest_int('p', 1, 2)
        
        model = KNeighborsClassifier(
            n_neighbors=knn_n_neighbors,
            weights=knn_weights,
            algorithm=knn_algorithm,
            leaf_size=knn_leaf_size,
            p=knn_p
        )

    elif classifier == 'DecisionTree':
        # DecisionTree params
        dt_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        dt_splitter = trial.suggest_categorical('splitter', ['best', 'random'])
        
        model = DecisionTreeClassifier(
            random_state=SEED,
            criterion=dt_criterion,
            splitter=dt_splitter
        )

    elif classifier == 'SVM':
        # SVM params
        svm_C = trial.suggest_categorical('svm_C', [0.1, 1, 10, 100, 1000])
        svm_degree = trial.suggest_categorical('svm_degree', [0, 1, 2, 3, 4, 5, 6])
        
        model = SVC(
            random_state=SEED,
            C=svm_C,
            degree=svm_degree
        )

    elif classifier == 'RandomForest':
        # RandomForest params
        rf_max_depth = trial.suggest_categorical('rf_max_depth', [80, 90, 100, 110])
        rf_max_features = trial.suggest_categorical('rf_max_features', [2, 3])
        rf_min_samples_leaf = trial.suggest_categorical('rf_min_sample_leaf', [8, 10, 12])
        rf_n_estimators = trial.suggest_categorical('rf_n_estimators', [100, 200, 300, 1000])
        
        model = RandomForestClassifier(
            random_state=SEED,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_leaf=rf_min_samples_leaf,
            n_estimators=rf_n_estimators
        )

    elif classifier == 'LightGBM':
        # LightGBM params
        lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 20, 200)
        lgbm_learning_rate = trial.suggest_categorical('lgbm_learning_rate', [0.01, 0.05, 0.1])
        lgbm_num_leaves = trial.suggest_categorical('lgbm_num_leaves', [80, 100, 150, 200])
#         lgbm_boosting_type = trial.suggest_categorical('lgbm_boosting_type', ['gbdt', 'dart', 'goss', 'rf'])
        lgbm_subsample = trial.suggest_categorical('lgbm_subsample', [1, 0.8, 0.7, 0.5])
        
        model = lgb.LGBMClassifier(
            random_state=SEED,
            max_depth=lgbm_max_depth,
            num_leaves=lgbm_num_leaves,
            learning_rate=lgbm_learning_rate,
            subsample=lgbm_subsample,
#             boosting_type=lgbm_boosting_type
        )

    elif classifier == 'Xgboost':
        # Xgboost params
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 10, 200)
        xgb_learning_rate = trial.suggest_categorical('xgb_learning_rate', [0.01, 0.05, 0.1])
#         xgb_booster = trial.suggest_categorical('xgb_boosting_type', ['gbdt', 'dart', 'goss', 'rf'])
        
        model = xgb.XGBClassifier(
            random_state=SEED,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
#             booster=xgb_booster
        )
        model
    else:
        return
        
    return cross_val_score(model, X, y, n_jobs=-1, scoring='accuracy', cv=kf).mean()


# In[ ]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=400)

trial = study.best_trial
print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


# In[54]:


# Test Data Laebeling

kf = KFold(n_splits=8, random_state=SEED, shuffle=True)

model = lgb.LGBMClassifier(random_state=SEED, max_depth=81, learning_rate=0.1, num_leaves=200, subsample=0.7)

for train_index, test_index in kf.split(X):
    train_X, test_X = X.iloc[train_index], X.iloc[test_index]
    train_y, test_y = y.iloc[train_index], y.iloc[train_index]

    model.fit(train_X, train_y)

pr = model.predict(test_data)
pr


# In[57]:


result = pd.DataFrame(pr)
result


# In[58]:


result.to_csv(f'./data/lgbm_dep_21_subsample_1.csv', index=False)


# In[ ]:




