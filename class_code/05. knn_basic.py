# KNN basic Example

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)
print(iris_X.shape)   # (150, 4)

# scaling input data 

scaler = StandardScaler()
scaler.fit(iris_X)
iris_X = scaler.transform(iris_X)


# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(iris_X, iris_y, test_size=0.3,\
                     random_state=1234) 

# Define learning model
model =  KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(train_X, train_y)

# Make predictions using the testing set
pred_y = model.predict(test_X)
print(pred_y)

# model evaluation: accuracy #############

acc = accuracy_score(test_y, pred_y)
print('Accuracy : {0:3f}'.format(acc))
