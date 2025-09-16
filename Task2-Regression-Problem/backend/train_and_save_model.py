import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Data Loading ---
# This script loads the actual House Prices training data from train.csv.
data_dir = 'data'
data_path = os.path.join(data_dir, 'train.csv')
try:
    df = pd.read_csv(data_path)
    print(f"Successfully loaded '{data_path}'")
except FileNotFoundError:
    print(f"Error: '{data_path}' not found. Please create the 'data' directory inside 'backend' and place the downloaded 'train.csv' file there.")
    exit()

# --- Step 2: Feature Engineering and Preprocessing ---
# Select features from the dataset that will be used for training.
features = ['GrLivArea', 'GarageCars', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', '1stFlrSF', '2ndFlrSF']
target = 'SalePrice'

# Perform Feature Engineering: Create a new feature from existing ones.
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
features.append('TotalSF')

# Use a simplified set of features for the frontend form.
frontend_features = ['GrLivArea', 'GarageCars', 'OverallQual', 'TotalBsmtSF', 'YearBuilt']

# Handle missing values for the selected features.
imputer = SimpleImputer(strategy='constant', fill_value=0)
df[features] = imputer.fit_transform(df[features])

# --- Step 3: Model Pipeline Creation and Training ---
# Define a robust pipeline for preprocessing and modeling.
def get_model_pipeline(regressor):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])

# Initialize the models for comparison.
models = {
    'LinearRegression': get_model_pipeline(LinearRegression()),
    'RandomForest': get_model_pipeline(RandomForestRegressor(n_estimators=100, random_state=42)),
    'XGBoost': get_model_pipeline(XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
}

# Use K-Fold Cross-Validation to evaluate each model.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_rmse = float('inf')

print("\nEvaluating models with K-Fold Cross-Validation (5 splits)...")
for name, model in models.items():
    # Cross-validation returns a list of scores for each fold.
    scores = cross_val_score(model, df[features], df[target], scoring='neg_root_mean_squared_error', cv=kf)
    rmse_scores = -scores  # Convert negative scores to positive RMSE.
    mean_rmse = np.mean(rmse_scores)
    
    print(f"Model: {name}")
    print(f"  RMSE Scores (per fold): {rmse_scores}")
    print(f"  Mean RMSE: ${mean_rmse:,.2f}")

    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_model = model

# Train the best model on the entire dataset and save it.
print(f"\nTraining the best model ({type(best_model.named_steps['regressor']).__name__}) on the full dataset...")
best_model.fit(df[features], df[target])
print("Model training complete.")

# --- Step 4: Save the Trained Model ---
# Ensure the models directory exists before saving the file.
model_dir = 'src/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# Save the best trained pipeline to a file using joblib.
model_path = os.path.join(model_dir, 'house_price_model.pkl')
joblib.dump(best_model, model_path)
print(f"Best model saved successfully to: {model_path}")

# --- Step 5: Generate and Save Visualizations ---
# This code will generate the charts and save them as PNG files.
reports_dir = '../frontend/public/reports'
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)
    print(f"Created directory: {reports_dir}")

print("\nGenerating visualizations...")
# Plot 1: Overall Quality vs. SalePrice Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='OverallQual', y='SalePrice', data=df)
plt.title('Overall Quality vs. SalePrice')
plt.savefig(os.path.join(reports_dir, 'overallqual_vs_saleprice.png'))
plt.close()

# Plot 2: GrLivArea vs. SalePrice Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('GrLivArea vs. SalePrice')
plt.savefig(os.path.join(reports_dir, 'grlivarea_vs_saleprice.png'))
plt.close()

# Plot 3: Feature Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df[frontend_features + [target]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(reports_dir, 'feature_correlation_heatmap.png'))
plt.close()

# Plot 4: SalePrice Distribution Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title('SalePrice Distribution')
plt.savefig(os.path.join(reports_dir, 'saleprice_distribution.png'))
plt.close()

# Plot 5: Pie Chart for a Categorical Feature (example: GarageCars)
plt.figure(figsize=(8, 8))
garage_counts = df['GarageCars'].value_counts()
plt.pie(garage_counts, labels=garage_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Garage Car Capacity Distribution')
plt.savefig(os.path.join(reports_dir, 'garage_cars_pie_chart.png'))
plt.close()

print("Visualizations saved successfully.")

# --- Step 6: Example Prediction with the Best Model ---
sample_house_df = pd.DataFrame([{'GrLivArea': 1500, 'GarageCars': 2, 'OverallQual': 7, 'TotalBsmtSF': 1000, 'YearBuilt': 2005, '1stFlrSF': 1000, '2ndFlrSF': 500, 'TotalSF': 2500}])
prediction = best_model.predict(sample_house_df)
print(f"\nSample prediction with the best model: ${prediction[0]:,.2f}")