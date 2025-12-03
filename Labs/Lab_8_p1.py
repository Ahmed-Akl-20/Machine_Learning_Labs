from sklearn.decomposition import PCA

import numpy as np

store_data = np.array([
     [50, 200, 4.5],
     [30, 120, 3.8],
     [45, 180, 4.2], 
     [60, 210, 4.8]
])
# PCA

pca = PCA(n_components=3)
store_pca = pca.fit_transform(store_data)

#Print results
print("Original Store Data:")
print("Sales($k) Customers Rating")
print(store_data)

print("\nData after PCA:")
print(store_pca)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print(f"\nCumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")