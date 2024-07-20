
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the preprocessed dataset
data = pd.read_csv('preprocessed_dataset.csv')

# Extract relevant features for clustering
features = data.drop(columns=['Churn_Yes'])

# Determine the optimal number of clusters using the elbow method
sample_data = features.sample(frac=0.1, random_state=42)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(sample_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Train the K-Means model with the optimal number of clusters (e.g., 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(features)
data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(data['MonthlyCharges'], data['tenure'], c=data['Cluster'], cmap='viridis')
plt.xlabel('MonthlyCharges')
plt.ylabel('tenure')
plt.title('Clusters of Customers')
plt.show()

# Save the dataset with cluster labels
data.to_csv('preprocessed_dataset_with_clusters.csv', index=False)
