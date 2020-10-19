# Decision Tree Example


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import pydot             # need to install

# prepare the credit dataset
df = pd.read_csv('D:/data/liver.csv')
print(df.head())    
print(df.columns)   # column names

df_X = df.loc[:, df.columns != 'category']
df_y = df['category']

# Split the data into training/testing sets
train_X, test_X, train_y, test_y = \
    train_test_split(df_X, df_y, test_size=0.3,\
                     random_state=1234) 

# Define learning model (basic) #####################################
model =  DecisionTreeClassifier(random_state=1234)

# Train the model using the training sets
model.fit(train_X, train_y)

# performance evaluation
print('Train accuracy :', model.score(train_X, train_y))
print('Test accuracy :', model.score(test_X, test_y))

pred_y = model.predict(test_X)
confusion_matrix(test_y, pred_y)

# Define learning model (tuening) #####################################
model =  DecisionTreeClassifier(max_depth=4, random_state=1234)

# Train the model using the training sets
model.fit(train_X, train_y)

# performance evaluation
print('Train accuracy :', model.score(train_X, train_y))
print('Test accuracy :', model.score(test_X, test_y))

pred_y = model.predict(test_X)
confusion_matrix(test_y, pred_y)

# visualize tree
export_graphviz(model, out_file='tree_model.dot', feature_names = train_X.columns,
                class_names = 'category',
                rounded = True, proportion = False, precision = 2, filled = True)
(graph,) = pydot.graph_from_dot_file('tree_model.dot', encoding='UTF-8')
graph.write_png('decision_tree.png')   # save tree image

#from IPython.display import Image
#Image(filename = 'decision_tree.png')