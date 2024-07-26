
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fpdf import FPDF

# Load the dataset
file_path = 'Dataset (ATS)-1.csv'
df = pd.read_csv(file_path)

# Initial check and visualization of missing data
print("First few rows of the dataset:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Handle missing data by filling missing values in numerical columns with their mean
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Encode categorical variables and remove duplicates
categorical_columns = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                       'Contract', 'Churn']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df_encoded.drop_duplicates(inplace=True)

# Split the dataset into features and target
X = df_encoded.drop('Churn_Yes', axis=1, errors='ignore')
y = df_encoded['Churn_Yes']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling techniques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaled data and labels into CSV files
os.makedirs('Data_Preparation', exist_ok=True)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled_df.to_csv('Data_Preparation/X_train_scaled.csv', index=False)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_test_scaled_df.to_csv('Data_Preparation/X_test_scaled.csv', index=False)
y_train.to_csv('Data_Preparation/y_train.csv', index=False)
y_test.to_csv('Data_Preparation/y_test.csv', index=False)

# Determine optimal number of clusters using the elbow method and visualize
sample_data = X_train_scaled_df.sample(frac=0.1, random_state=42)  # Sample data for elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(sample_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='1', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Clustering with the optimal number of clusters
k_optimal = 2  # Default start if no significant elbow is found
for i in range(1, len(wcss)-1):
    if wcss[i-1] - wcss[i] < (wcss[i] - wcss[i+1]):
        k_optimal = i +1
        break

kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
kmeans.fit(X_train_scaled_df)
X_train_scaled_df['Cluster'] = kmeans.labels_+1

# Visualize the clusters using scatter plot
# Here we choose two features for simplicity; you can change these based on your dataset
plt.scatter(X_train_scaled_df['MonthlyCharges'], X_train_scaled_df['tenure'], c=X_train_scaled_df['Cluster'], cmap='viridis')
plt.xlabel('Monthly Charges (scaled)')
plt.ylabel('Tenure (scaled)')
plt.title(f'Clusters of Customers - {k_optimal} Clusters')
plt.colorbar(label='Cluster')
plt.show()

# Save the final dataset with clusters
X_train_scaled_df.to_csv('Data_Preparation/final_preprocessed_and_clustered_data.csv', index=False)
print(f'Optimal number of clusters determined: {k_optimal}')

# Create a PDF document explaining scaling techniques
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Scaling Techniques Documentation', 0, 1, 'C')
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, '''
Scaling Techniques Applied:
We applied StandardScaler to normalize the numerical features of the dataset. StandardScaler
 standardizes features by removing the mean and scaling to unit variance.
''')
pdf.output('Data_Preparation/Scaling_Techniques.pdf')
