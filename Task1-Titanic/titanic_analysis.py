# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib  # For saving the model

# Load the data
train_df = pd.read_csv('E:/BT-AI-ML-SEP-2025-114/Task1-Titanic/data/train.csv')
test_df = pd.read_csv('E:/BT-AI-ML-SEP-2025-114/Task1-Titanic/data/test.csv')

# Data Exploration
print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("\nTraining Data Info:")
train_df.info()
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

# Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=train_df, ax=axes[0, 0])
axes[0, 0].set_title('Survival by Gender')

# Survival by Pclass
sns.countplot(x='Survived', hue='Pclass', data=train_df, ax=axes[0, 1])
axes[0, 1].set_title('Survival by Passenger Class')

# Age distribution
train_df['Age'].hist(bins=30, ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution')

# Fare distribution
train_df['Fare'].hist(bins=30, ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution')

plt.tight_layout()
plt.savefig('E:/BT-AI-ML-SEP-2025-114/Task1-Titanic/titanic_eda.png')
plt.show()

# Data Preprocessing
def preprocess_data(df):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Extract title from Name
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Replace rare titles with more common ones
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
    
    # Convert Sex to numerical values
    df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})
    
    # Fill missing Age values with median age by Title
    df_processed['Age'].fillna(df_processed.groupby('Title')['Age'].transform('median'), inplace=True)
    
    # Create Age groups
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=[0, 12, 20, 40, 120], labels=['Child', 'Teen', 'Adult', 'Elderly'])
    
    # Fill missing Embarked values with mode
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    
    # Convert Embarked to numerical values
    df_processed['Embarked'] = df_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Fill missing Fare values with median
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    
    # Create Fare groups
    df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], 4, labels=[1, 2, 3, 4])
    
    # Create FamilySize feature
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    
    # Create IsAlone feature
    df_processed['IsAlone'] = 1
    df_processed['IsAlone'].loc[df_processed['FamilySize'] > 1] = 0
    
    # Drop unnecessary columns
    df_processed = df_processed.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare'], axis=1)
    
    # Convert categorical variables to dummy variables
    df_processed = pd.get_dummies(df_processed, columns=['Title', 'AgeGroup', 'FareGroup'], drop_first=True)
    
    return df_processed

# Preprocess the training and test data
train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)

# Ensure both datasets have the same columns
missing_cols = set(train_processed.columns) - set(test_processed.columns)
for col in missing_cols:
    test_processed[col] = 0

# Reorder test columns to match train
test_processed = test_processed[train_processed.columns.drop('Survived')]

# Split the training data into features and target
X = train_processed.drop('Survived', axis=1)
y = train_processed['Survived']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --------------------------------------------------------------------------------
# HYPERPARAMETER TUNING FOR DECISION TREE (NEW)
# ---------------------------------------------------------------------------------
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                          param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters for Decision Tree:", grid_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))

# Get the best model
best_dtree = grid_search.best_estimator_

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MODEL TRAINING AND EVALUATION
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

# Logistic Regression Model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_val_scaled)
y_pred_prob_logreg = logreg.predict_proba(X_val_scaled)[:, 1]  # For ROC curve

print("Logistic Regression Accuracy: {:.4f}".format(accuracy_score(y_val, y_pred_logreg)))
print("Logistic Regression Classification Report:")
print(classification_report(y_val, y_pred_logreg))

# Decision Tree Model (with best parameters)
best_dtree.fit(X_train, y_train)
y_pred_dtree = best_dtree.predict(X_val)
y_pred_prob_dtree = best_dtree.predict_proba(X_val)[:, 1]  # For ROC curve

print("Optimized Decision Tree Accuracy: {:.4f}".format(accuracy_score(y_val, y_pred_dtree)))
print("Decision Tree Classification Report:")
print(classification_report(y_val, y_pred_dtree))

# *****************************************************************************
# CROSS-VALIDATION (NEW)
# *****************************************************************************
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)

# Cross-validation for Logistic Regression
lr_cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Logistic Regression CV Accuracy: {:.4f} (+/- {:.4f})".format(lr_cv_scores.mean(), lr_cv_scores.std() * 2))

# Cross-validation for Decision Tree
dt_cv_scores = cross_val_score(best_dtree, X_train, y_train, cv=5, scoring='accuracy')
print("Decision Tree CV Accuracy: {:.4f} (+/- {:.4f})".format(dt_cv_scores.mean(), dt_cv_scores.std() * 2))

# =============================================================================
# ROC CURVE AND AUC (NEW)
# =============================================================================
print("\n" + "="*50)
print("ROC CURVE ANALYSIS")
print("="*50)

# Calculate ROC curve and AUC for both models
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_pred_prob_logreg)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_dt, tpr_dt, _ = roc_curve(y_val, y_pred_prob_dtree)
roc_auc_dt = auc(fpr_dt, tpr_dt)

print("Logistic Regression AUC: {:.4f}".format(roc_auc_lr))
print("Decision Tree AUC: {:.4f}".format(roc_auc_dt))

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='Logistic Regression (AUC = {:.4f})'.format(roc_auc_lr))
plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label='Decision Tree (AUC = {:.4f})'.format(roc_auc_dt))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('E:/BT-AI-ML-SEP-2025-114/Task1-Titanic/roc_curve.png')
plt.show()

# -----------------------------------------------------------------------------
# FEATURE IMPORTANCE AND CONFUSION MATRIX
# -----------------------------------------------------------------------------

# Feature Importance for Decision Tree
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_dtree.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.heatmap(confusion_matrix(y_val, y_pred_logreg), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Logistic Regression Confusion Matrix')

sns.heatmap(confusion_matrix(y_val, y_pred_dtree), annot=True, fmt='d', ax=axes[1], cmap='Blues')
axes[1].set_title('Decision Tree Confusion Matrix')

plt.tight_layout()
plt.savefig('E:/BT-AI-ML-SEP-2025-114/Task1-Titanic/confusion_matrix.png')
plt.show()

# =============================================================================
# SAVE THE BEST MODEL (NEW)
# =============================================================================
joblib.dump(best_dtree, 'best_decision_tree_model.pkl')
print("\nBest model saved as 'best_decision_tree_model.pkl'")

# *******************************************************************************
# FINAL SUMMARY
# *******************************************************************************
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print("Best Model: Decision Tree with parameters", grid_search.best_params_)
print("Validation Accuracy: {:.4f}".format(accuracy_score(y_val, y_pred_dtree)))
print("Cross-Validation Accuracy: {:.4f} (+/- {:.4f})".format(dt_cv_scores.mean(), dt_cv_scores.std() * 2))
print("AUC Score: {:.4f}".format(roc_auc_dt))