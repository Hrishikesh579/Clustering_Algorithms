import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load the data from Kaggle
data = pd.read_csv('./datasets/marketing_campaign.csv', sep='\t', engine='python')

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Impute missing values for categorical and numerical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# For numerical columns, use the mean
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
num_imputer = SimpleImputer(strategy='mean')
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# Convert categorical columns to numeric using one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Separate features and target variable
X = data_encoded.iloc[:, :-1]  # Assuming the last column is the target
y = data_encoded.iloc[:, -1]

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

optimal_k = 13  # Replace with the value found from the elbow method
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Predicting the clusters
y_pred = kmeans.predict(X_scaled)  # Predicting on the scaled data

# Using PCA to reduce dimensions for visualization
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)  # Apply PCA to the scaled features

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('K-means Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()