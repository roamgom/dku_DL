# kmeans example

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [4, 3], [2, 5],
              [8, 5], [10, 6], [9, 4]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# cluster label
kmeans.labels_
# bind data & cluster label
np.hstack((X, kmeans.labels_.reshape(-1, 1)))   

# center of clusters
kmeans.cluster_centers_

# predict new data
kmeans.predict([[0, 0], [12, 3]])


