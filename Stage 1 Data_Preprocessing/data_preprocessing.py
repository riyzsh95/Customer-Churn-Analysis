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
