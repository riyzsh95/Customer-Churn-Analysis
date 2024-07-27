import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# Stage 2: Feature Engineering and Scaling
# ============================================================

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
