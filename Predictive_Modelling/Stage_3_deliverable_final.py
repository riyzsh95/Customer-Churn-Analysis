import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fpdf import FPDF
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Load the dataset
file_path = 'Dataset (ATS)-1.csv'  # Path to the dataset file
df = pd.read_csv(file_path)  # Load the dataset into a pandas DataFrame

# Initial check and visualization of missing data
print("First few rows of the dataset:")
print(df.head())  # Display the first few rows of the dataset
print("\nMissing values per column:")
print(df.isnull().sum())  # Check for missing values in each column
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')  # Visualize missing data using a heatmap
plt.title('Missing Data Heatmap')
plt.show()

# Handle missing data by filling missing values in numerical columns with their mean
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select numerical columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())  # Fill missing values with column mean

# Encode categorical variables and remove duplicates
categorical_columns = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                       'Contract', 'Churn']  # List of categorical columns to encode
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)  # One-hot encode categorical variables
df_encoded.drop_duplicates(inplace=True)  # Remove duplicate rows

# Split the dataset into features and target
X = df_encoded.drop('Churn_Yes', axis=1, errors='ignore')  # Features (exclude target column)
y = df_encoded['Churn_Yes']  # Target (Churn label)

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data

# Apply scaling techniques
scaler = StandardScaler()  # Initialize the StandardScaler
X_train_scaled = scaler.fit_transform(X_train)  # Scale the training data
X_test_scaled = scaler.transform(X_test)  # Scale the testing data

# Save the scaled data and labels into CSV files
os.makedirs('Data_Preparation', exist_ok=True)  # Create directory if it doesn't exist
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # Convert scaled data to DataFrame
X_train_scaled_df.to_csv('Data_Preparation/X_train_scaled.csv', index=False)  # Save scaled training data to CSV
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)  # Convert scaled test data to DataFrame
X_test_scaled_df.to_csv('Data_Preparation/X_test_scaled.csv', index=False)  # Save scaled test data to CSV
y_train.to_csv('Data_Preparation/y_train.csv', index=False)  # Save training labels to CSV
y_test.to_csv('Data_Preparation/y_test.csv', index=False)  # Save test labels to CSV

# Determine optimal number of clusters using the elbow method and visualize
sample_data = X_train_scaled_df.sample(frac=0.1, random_state=42)  # Take a sample of the data for clustering
wcss = []  # List to store within-cluster sum of squares for each k
for i in range(1, 11):  # Test k values from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # Initialize KMeans with k clusters
    kmeans.fit(sample_data)  # Fit KMeans to the sample data
    wcss.append(kmeans.inertia_)  # Append WCSS to the list

# Plot the WCSS to visualize the elbow point
plt.plot(range(1, 11), wcss, marker='1', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Clustering with the optimal number of clusters
k_optimal = 2  # Default start if no significant elbow is found
for i in range(1, len(wcss)-1):  # Iterate to find the elbow point
    if wcss[i-1] - wcss[i] < (wcss[i] - wcss[i+1]):  # Check for significant drop in WCSS
        k_optimal = i + 1  # Set optimal k to current index + 1
        break

kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)  # Initialize KMeans with optimal k
kmeans.fit(X_train_scaled_df)  # Fit KMeans to the entire scaled training data
X_train_scaled_df['Cluster'] = kmeans.labels_ + 1  # Assign cluster labels to the data

# Visualize the clusters using scatter plot
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
pdf = FPDF()  # Initialize PDF
pdf.add_page()  # Add a page
pdf.set_font('Arial', 'B', 12)  # Set font for the header
pdf.cell(0, 10, 'Scaling Techniques Documentation', 0, 1, 'C')  # Add header text
pdf.set_font('Arial', '', 12)  # Set font for the body
pdf.multi_cell(0, 10, '''
Scaling Techniques Applied:
We applied StandardScaler to normalize the numerical features of the dataset. StandardScaler
 standardizes features by removing the mean and scaling to unit variance.
''')  # Add body text
pdf.output('Data_Preparation/Scaling_Techniques.pdf')  # Save the PDF

# Load preprocessed and scaled data
X_train_scaled_df = pd.read_csv('Data_Preparation/X_train_scaled.csv')  # Load scaled training data
X_test_scaled_df = pd.read_csv('Data_Preparation/X_test_scaled.csv')  # Load scaled test data
y_train = pd.read_csv('Data_Preparation/y_train.csv').values.ravel()  # Load and flatten training labels
y_test = pd.read_csv('Data_Preparation/y_test.csv').values.ravel()  # Load and flatten test labels

# ---- Existing ANN Model ---- #

# Define the architecture of the existing ANN model
model = Sequential([  # Initialize the Sequential model
    Input(shape=(X_train_scaled_df.shape[1],)),  # Input layer with number of features
    Dense(64, activation='relu'),  # First hidden layer with 64 neurons and ReLU activation
    Dense(32, activation='relu'),  # Second hidden layer with 32 neurons and ReLU activation
    Dense(128, activation='relu'),  # Third hidden layer with 128 neurons and ReLU activation
    Dropout(0.3),  # Dropout layer to prevent overfitting
    Dense(64, activation='relu'),  # Fourth hidden layer with 64 neurons and ReLU activation
    Dropout(0.3),  # Another Dropout layer
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model with Adam optimizer

# Define learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)  # Reduce LR on plateau

# Define early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Early stopping

# Train the existing ANN model
history = model.fit(X_train_scaled_df, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)  # Train the model

# Evaluate the existing ANN model
loss, accuracy = model.evaluate(X_test_scaled_df, y_test, verbose=0)  # Evaluate the model on test data
print(f'\nTest Accuracy (Existing ANN): {accuracy:.4f}')
print(f'Test Loss (Existing ANN): {loss:.4f}')

# Store accuracy of the existing ANN model in model_performance dictionary
model_performance = {}  # Initialize a dictionary to store the performance of each model
model_performance['Existing ANN'] = accuracy

# Predict customer churn using the existing ANN model
y_pred_ann = (model.predict(X_test_scaled_df) > 0.5).astype(int)  # Make predictions and apply threshold

# Classification report and confusion matrix for the existing ANN model
print("\nClassification Report (Existing ANN):")
print(classification_report(y_test, y_pred_ann))  # Print the classification report
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)  # Generate the confusion matrix

# Plot confusion matrix for the existing ANN model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ann, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])  # Plot the confusion matrix
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Existing ANN)')
plt.show()

# Save the existing ANN model in Keras format
model.save('my_model_existing_ann.keras')  # Save the trained model to a file

# ---- Logistic Regression Model ---- #

# Logistic Regression model definition and training
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predict using Logistic Regression
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Calculate accuracy for Logistic Regression and store in model_performance
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
model_performance['Logistic Regression'] = log_reg_accuracy
print(f"\nLogistic Regression Accuracy: {log_reg_accuracy:.4f}")

# Classification report and confusion matrix for Logistic Regression
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log_reg))
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)

# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

# ---- Deep Neural Network (DNN) Model ---- #

# Define a more complex DNN architecture
dnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # First hidden layer
    Dropout(0.4),  # Dropout to prevent overfitting
    Dense(64, activation='relu'),  # Second hidden layer
    Dropout(0.3),  # Another Dropout layer
    Dense(32, activation='relu'),  # Third hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the DNN model
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the DNN model

# Train the DNN model
dnn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)  # Train the DNN model

# Evaluate the DNN model
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test_scaled, y_test, verbose=0)  # Evaluate the DNN model
model_performance['Deep Neural Network'] = dnn_accuracy  # Store the accuracy in model_performance

print(f"\nDeep Neural Network Accuracy: {dnn_accuracy:.4f}")

# Predict using the DNN model
y_pred_dnn = (dnn_model.predict(X_test_scaled) > 0.5).astype(int)

# Classification report and confusion matrix for the DNN model
print("\nClassification Report (Deep Neural Network):")
print(classification_report(y_test, y_pred_dnn))  # Print the classification report
conf_matrix_dnn = confusion_matrix(y_test, y_pred_dnn)  # Generate the confusion matrix

# Plot confusion matrix for the DNN model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Deep Neural Network)')
plt.show()

# ---- Compare All Models ---- #
# Identify and print the best model based on accuracy
best_model_name = max(model_performance, key=model_performance.get)
best_model_accuracy = model_performance[best_model_name]
print(f"\nBest Model: {best_model_name} with an accuracy of {best_model_accuracy:.4f}")