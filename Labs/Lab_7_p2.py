from sklearn.cluster import KMeans
import numpy as np

data = np.array([

[2, 3], [3, 3], [6, 5], [8, 8], [3,4],

[5, 2], [7, 6], [4, 7], [9, 4], [8, 2],

[1, 1], [3, 7], [6, 8], [4, 5], [7, 3]

])

# عدد المجموعات K = 3 

kmeans = KMeans (n_clusters=3)

#تدريب النموذج 

kmeans.fit(data)

#النتايج  

centroids = kmeans.cluster_centers_
labels = kmeans.labels_



print("Centroids:")
print(centroids)
print("Labels:")
print(labels)