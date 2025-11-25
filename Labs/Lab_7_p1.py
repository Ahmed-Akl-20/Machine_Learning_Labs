from sklearn.cluster import KMeans
import numpy as np


#Sample data
data= np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

#Specify the number of clusters (K) 
kmeans = KMeans (n_clusters=2)


#Fit the date to the algorithm
kmeans.fit(data)

#Get the cluster centroids and labels
centroids = kmeans.cluster_centers_

labels = kmeans.labels_

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)