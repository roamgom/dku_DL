# KNN kfold Example

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)

# Define fold

kf = KFold(n_splits=5, random_state=123, shuffle=True)       # 5 fold

# Define learning model
model =  KNeighborsClassifier(n_neighbors=3)

acc = np.zeros(5)     # 5 fold
i = 0                 # fold no

for train_index, test_index in kf.split(iris_X):
    print("fold:", i)
    
    train_X, test_X = iris_X[train_index], iris_X[test_index]
    train_y, test_y =  iris_y[train_index], iris_y[test_index]

    # Train the model using the training sets
    model.fit(train_X, train_y)

    # Make predictions using the testing set
    pred_y = model.predict(test_X)
    #print(pred_y)

    # model evaluation: accuracy #############
    acc[i] = accuracy_score(test_y, pred_y)
    print('Accuracy : {0:3f}'.format(acc[i]))
    i += 1

print("5 fold :", acc)
print("mean accuracy :", np.mean(acc))

