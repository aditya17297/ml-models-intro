from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

## import data set

dataset = pd.read_csv('/Users/adityaagrawal/PycharmProjects/PythonProject/4_ClusteringMLModels/1_KMeanClustering/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

## using dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()


## Training the k_means in 5 clusters
from  sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_kmeans = model.fit_predict(X) ## fit_transform will train and also predict the classification/cluster
print(y_kmeans)

## Visualise
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of dataset')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

