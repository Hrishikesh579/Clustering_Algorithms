import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
# Load the data from Kaggle
data = pd.read_csv('./datasets/marketing_campaign.csv', sep='\t', engine='python')

# Separate features and target variable
X = data.iloc[:,9:14]  # Assuming the last column is the target
y = data.iloc[:, 14]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform KMeans clustering
WCS = []
for k in range(1, 22):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    WCS.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 22), WCS)
plt.xticks(range(1, 22))
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Graph value of K')
plt.show()

optimal_k = 2  # Replace with the value found from the elbow method
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Predicting the clusters
y_pred = kmeans.predict(X_scaled)  # Predicting on the scaled data

# Using PCA to reduce dimensions for visualization
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled) 

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('K-means Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# printing Silhoette score for checking efficiency of the model
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score for {optimal_k} clusters: {silhouette_avg:.2f}')
