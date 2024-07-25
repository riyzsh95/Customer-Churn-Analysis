
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Preprocessed Dataset from Stage 1
file_path_stage2 = 'preprocessed_dataset.csv'
data_encoded = pd.read_csv(file_path_stage2)

# Display the first few rows to verify
print("\nFirst few rows of the preprocessed dataset:")
print(data_encoded.head())

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

# Display the selected features
print("\nSelected features for analysis:")
print(data_selected.head())

# Engineer New Features or Transform Existing Ones

# Create new features: Tenure groups and MonthlyCharges bins
data_encoded['TenureGroup'] = pd.cut(
    data_encoded['tenure'],
    bins=[0, 12, 24, 48, 72, 1000],
    labels=['0-1yr', '1-2yrs', '2-4yrs', '4-6yrs', '6+yrs']
)

data_encoded['MonthlyChargesBin'] = pd.cut(
    data_encoded['MonthlyCharges'],
    bins=[0, 30, 60, 90, 120],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Display the new features
print("\nEngineered features:")
print(data_encoded[['TenureGroup', 'MonthlyChargesBin']].head())

# Handle Feature Scaling and Normalization

# Define the features to scale
features_to_scale = ['tenure', 'MonthlyCharges']

# Standardize the features
scaler = StandardScaler()
data_encoded[features_to_scale] = scaler.fit_transform(data_encoded[features_to_scale])

# Display the scaled features
print("\nScaled features:")
print(data_encoded[features_to_scale].head())

# Save the Updated Dataset and Document Your Process

# Save the preprocessed dataset for Stage 2
data_encoded.to_csv('preprocessed_dataset_stage2.csv', index=False)
print("\nPreprocessed dataset for Stage 2 saved as 'preprocessed_dataset_stage2.csv'.")

# Document the feature engineering steps
with open('feature_engineering_log.txt', 'w') as f:
    f.write("Feature Engineering Steps:\n")
    f.write("1. Selected relevant features for analysis.\n")
    f.write("2. Created new features: Tenure groups and MonthlyCharges bins.\n")
    f.write("3. Standardized numerical features: tenure, MonthlyCharges.\n")
    f.write("4. Saved the preprocessed data for model training.\n")

print("Feature engineering steps documented in 'feature_engineering_log.txt'.")


# Data Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Dataset (ATS)-1.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check and display missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Visualize missing data
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Handle missing data: Fill missing values in numerical columns with mean
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Display the dataset after handling missing values
print("\nData after handling missing values:")
print(data.head())

# Encode categorical variables
categorical_columns = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'Churn']
# Ensure only existing categorical columns are encoded
existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
data_encoded = pd.get_dummies(data, columns=existing_categorical_columns, drop_first=True)

# Remove duplicates
data_encoded = data_encoded.drop_duplicates()

# Ensure proper data types
if 'tenure' in data_encoded.columns:
    data_encoded['tenure'] = data_encoded['tenure'].astype(int)

if 'MonthlyCharges' in data_encoded.columns:
    data_encoded['MonthlyCharges'] = data_encoded['MonthlyCharges'].astype(float)

# Save the preprocessed dataset
data_encoded.to_csv('preprocessed_dataset.csv', index=False)

print("\nPreprocessed dataset saved as 'preprocessed_dataset.csv'.")


# Clustering Analysis Script

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


# Clustering Analysis Complete

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
