from sklearn.decomposition import PCA
import numpy as np

#Student data
student_data = np.array([
     [90, 70], 
     [60, 80], 
     [75, 95], 
     [95, 85] 
])

pca = PCA(n_components=2)

student_pca = pca.fit_transform(student_data)

print("Original Data:") 
print(student_data)

print("\nData after PCA:")
print(student_pca)

print("\nExplained variance:") 
print(pca.explained_variance_ratio_)

print(f"Total: {sum(pca.explained_variance_ratio_):.2%}")
