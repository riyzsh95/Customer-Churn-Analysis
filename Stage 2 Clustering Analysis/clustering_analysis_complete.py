
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the preprocessed dataset
file_path_stage1 = 'preprocessed_dataset.csv'
data_encoded = pd.read_csv(file_path_stage1)

# Ensure all necessary categorical columns are present
required_features = [
    'tenure', 'MonthlyCharges',
    'InternetService_Fiber optic', 'PhoneService_Yes', 'MultipleLines_Yes',
    'Contract_Two year', 'gender_Male', 'Dependents_Yes', 'Churn_Yes'
]

# Check if all required features are present after encoding
missing_features = [feature for feature in required_features if feature not in data_encoded.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

# Select the relevant features
data_selected = data_encoded[required_features]

# Create new features: Tenure groups and MonthlyCharges bins
data_selected['TenureGroup'] = pd.cut(
    data_selected['tenure'],
    bins=[0, 12, 24, 48, 72, 1000],
    labels=['0-1yr', '1-2yrs', '2-4yrs', '4-6yrs', '6+yrs']
)

data_selected['MonthlyChargesBin'] = pd.cut(
    data_selected['MonthlyCharges'],
    bins=[0, 30, 60, 90, 120],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Standardize the numerical features
features_to_scale = ['tenure', 'MonthlyCharges']
scaler = StandardScaler()
data_selected[features_to_scale] = scaler.fit_transform(data_selected[features_to_scale])

# Encode categorical features
data_encoded_stage2 = pd.get_dummies(data_selected, columns=['TenureGroup', 'MonthlyChargesBin'], drop_first=True)

# Extract relevant features for clustering, excluding the target column 'Churn_Yes'
features_stage2 = data_encoded_stage2.drop(columns=['Churn_Yes'])

# Use a smaller sample to find the optimal number of clusters using the elbow method
sample_data_stage2 = features_stage2.sample(frac=0.1, random_state=42)

# Calculate WCSS for different numbers of clusters
wcss_stage2 = []
for i in range(1, 11):
    kmeans_stage2 = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_stage2.fit(sample_data_stage2)
    wcss_stage2.append(kmeans_stage2.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss_stage2, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Train the K-Means model with the optimal number of clusters (e.g., 4)
optimal_clusters = 4
kmeans_stage2 = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans_stage2.fit(features_stage2)
data_encoded_stage2['Cluster'] = kmeans_stage2.labels_

# Visualize the clusters (example using two features for simplicity)
plt.figure(figsize=(10, 5))
plt.scatter(data_encoded_stage2['MonthlyCharges'], data_encoded_stage2['tenure'], c=data_encoded_stage2['Cluster'], cmap='viridis')
plt.xlabel('MonthlyCharges')
plt.ylabel('tenure')
plt.title('Clusters of Customers')
plt.show()

# Save the dataset with cluster labels
data_encoded_stage2.to_csv('preprocessed_dataset_stage2_with_clusters.csv', index=False)
