# KNN hyper parameter tuning Example
# using validation_curve (for single parameter)

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the diabetes dataset
wine_X, wine_y = datasets.load_wine(return_X_y=True)
print(wine_X.shape)   # (178, 13)

# scaling input data 

scaler = StandardScaler()
scaler.fit(wine_X)
wine_X = scaler.transform(wine_X)

# hyper parameter tuning
param_range = np.array([1,2,3,4,5,6,7,8,9])

train_scores, test_scores = \
    validation_curve(KNeighborsClassifier(), wine_X, wine_y,
                     param_name="n_neighbors", param_range=param_range,
                     cv=10, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

#mpl.rcParams["font.family"] = 'DejaVu Sans'
plt.plot(param_range, train_scores_mean, label="Training score", color="r")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.plot(param_range, test_scores_mean,
             label="Cross-validation score", color="g")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.title("Validation Curve with KNN")
plt.xlabel("n_neighbors")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.show()

# find best parameter value
ch = np.where(test_scores_mean == np.amax(test_scores_mean))
print('Best n_neighbors :', param_range[ch])
print('Best accuracy :', test_scores_mean[ch])

