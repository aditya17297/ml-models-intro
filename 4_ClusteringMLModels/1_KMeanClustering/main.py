from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## import data set

dataset = pd.read_csv('/Users/adityaagrawal/PycharmProjects/PythonProject/4_ClusteringMLModels/1_KMeanClustering/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


## find the optimal number of clusters via elbow method
## run k means for clusters 1 to 10 and calculate wcss for each
from sklearn.cluster import KMeans
wcss  = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

## Training the k_means in 5 clusters
model = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = model.fit_predict(X) ## fit_transform will train and also predict the classification/cluster
print(y_kmeans)

## Visualise
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=300, c='yellow', label='Centroid')
plt.title('Clusters of dataset')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

