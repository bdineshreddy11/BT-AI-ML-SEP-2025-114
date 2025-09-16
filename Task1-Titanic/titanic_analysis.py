# Titanic Survival Prediction - Task 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the Titanic dataset"""
    print("🚢 Titanic Survival Prediction Analysis")
    print("=" * 50)
    
    # Load datasets
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        print("✅ Data loaded successfully")
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
    except FileNotFoundError:
        print("❌ Error: Dataset files not found. Please ensure 'train.csv' and 'test.csv' are in the 'data' folder.")
        exit()
    
    # Display basic info
    print("\n📊 Data Exploration:")
    print("Missing values in training data:")
    print(train_df.isnull().sum().sort_values(ascending=False))
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocess and engineer features"""
    print("\n🔧 Feature Engineering in progress...")
    
    # Combine data for consistent preprocessing
    combined_data = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)
    
    # 1. Extract 'Title' from 'Name'
    combined_data['Title'] = combined_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    combined_data['Title'] = combined_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                                           'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combined_data['Title'] = combined_data['Title'].replace('Mlle', 'Miss')
    combined_data['Title'] = combined_data['Title'].replace('Ms', 'Miss')
    combined_data['Title'] = combined_data['Title'].replace('Mme', 'Mrs')
    
    # 2. Create FamilySize and IsAlone
    combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1
    combined_data['IsAlone'] = (combined_data['FamilySize'] == 1).astype(int)
    
    # 3. Handle missing values
    combined_data['Age'].fillna(combined_data.groupby('Title')['Age'].transform('median'), inplace=True)
    combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0], inplace=True)
    combined_data['Fare'].fillna(combined_data['Fare'].median(), inplace=True)
    combined_data.drop('Cabin', axis=1, inplace=True)
    
    # 4. Binning and categorization
    combined_data['AgeGroup'] = pd.cut(combined_data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                     labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    combined_data['FareGroup'] = pd.qcut(combined_data['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # 5. Drop irrelevant columns
    combined_data.drop(['PassengerId', 'Name', 'Ticket', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1, inplace=True)
    
    # 6. Encode categorical features
    combined_data = pd.get_dummies(combined_data, columns=['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup'], 
                                 drop_first=True)
    
    # Separate data back into train and test sets
    X = combined_data.iloc[:len(train_df)]
    y = train_df['Survived']
    X_test_final = combined_data.iloc[len(train_df):]
    
    print("✅ Feature engineering completed")
    print(f"Final training features: {X.shape[1]}")
    
    return X, y, X_test_final, test_df

def train_and_evaluate_models(X, y):
    """Train and evaluate machine learning models"""
    print("\n" + "=" * 50)
    print("🤖 Model Training and Evaluation")
    print("=" * 50)
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {}
    results = []
    
    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    log_reg = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log_reg = log_reg.predict(X_val_scaled)
    
    log_reg_accuracy = accuracy_score(y_val, y_pred_log_reg)
    print(f"Accuracy: {log_reg_accuracy:.4f}")
    print(classification_report(y_val, y_pred_log_reg))
    
    # Cross-validation
    log_reg_cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Val Accuracy: {log_reg_cv_scores.mean():.4f} (±{log_reg_cv_scores.std():.4f})")
    
    models['Logistic Regression'] = log_reg
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': log_reg_accuracy,
        'CV Score': log_reg_cv_scores.mean()
    })
    
    # --- Decision Tree with Hyperparameter Tuning ---
    print("\n--- Decision Tree ---")
    dtree = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_dtree = grid_search.best_estimator_
    y_pred_dtree = best_dtree.predict(X_val)
    
    dtree_accuracy = accuracy_score(y_val, y_pred_dtree)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {dtree_accuracy:.4f}")
    print(classification_report(y_val, y_pred_dtree))
    
    # Cross-validation
    dtree_cv_scores = cross_val_score(best_dtree, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Val Accuracy: {dtree_cv_scores.mean():.4f} (±{dtree_cv_scores.std():.4f})")
    
    models['Decision Tree'] = best_dtree
    results.append({
        'Model': 'Decision Tree',
        'Accuracy': dtree_accuracy,
        'CV Score': dtree_cv_scores.mean()
    })
    
    # --- Model Comparison ---
    print("\n" + "=" * 50)
    print("📈 Model Comparison")
    print("=" * 50)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Determine best model
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_model = models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name}")
    
    return best_model, X_train, X_val, y_train, y_val, results_df, models

def create_visualizations(X_train, y_train, best_model, X_val, y_val, models, train_df):
    """Create visualizations for the analysis"""
    print("\n" + "=" * 50)
    print("📊 Generating Visualizations")
    print("=" * 50)
    
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # --- EDA Plots ---
    # 1. Survival by Passenger Class
    plt.figure(figsize=(8,6))
    sns.barplot(x='Pclass', y='Survived', data=train_df)
    plt.title('Survival by Passenger Class')
    plt.ylabel('Survival Rate')
    plt.tight_layout()
    plt.savefig('visualizations/survival_by_class.png', dpi=300)
    plt.close()
    
    # 2. Age Distribution with survival overlay
    plt.figure(figsize=(8,6))
    sns.histplot(train_df, x='Age', hue='Survived', multiple='stack', kde=False)
    plt.title('Age Distribution with Survival')
    plt.tight_layout()
    plt.savefig('visualizations/age_distribution.png', dpi=300)
    plt.close()
    
    # 3. Gender Survival Comparison
    plt.figure(figsize=(6,4))
    sns.barplot(x='Sex', y='Survived', data=train_df)
    plt.title('Survival by Gender')
    plt.tight_layout()
    plt.savefig('visualizations/gender_survival.png', dpi=300)
    plt.close()
    
    # 4. Correlation Heatmap (numeric columns only)
    plt.figure(figsize=(12,8))
    numeric_cols = train_df.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
    plt.close()
    
    # --- Feature Importance (Decision Tree) ---
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances)
        plt.title('Top 10 Feature Importance (Decision Tree)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300)
        plt.close()
    
    # --- Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Logistic Regression
    log_reg = models['Logistic Regression']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    y_pred_log_reg = log_reg.predict(X_val_scaled)
    sns.heatmap(confusion_matrix(y_val, y_pred_log_reg), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    # Decision Tree
    y_pred_dtree = best_model.predict(X_val)
    sns.heatmap(confusion_matrix(y_val, y_pred_dtree), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Decision Tree')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png', dpi=300)
    plt.close()
    
    # --- ROC Curve ---
    plt.figure(figsize=(8,6))
    y_prob_log_reg = log_reg.predict_proba(X_val_scaled)[:, 1]
    fpr_log, tpr_log, _ = roc_curve(y_val, y_prob_log_reg)
    plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc(fpr_log, tpr_log):.3f})')
    y_prob_dtree = best_model.predict_proba(X_val)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_val, y_prob_dtree)
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc(fpr_dt, tpr_dt):.3f})')
    plt.plot([0,1],[0,1],'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png', dpi=300)
    plt.close()

def save_model_and_predictions(best_model, X_test_final, test_df):
    """Save the best model and create predictions"""
    joblib.dump(best_model, 'best_titanic_model.pkl')
    final_predictions = best_model.predict(X_test_final)
    submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": final_predictions
    })
    submission.to_csv('titanic_predictions.csv', index=False)
    return submission

def main():
    train_df, test_df = load_and_explore_data()
    X, y, X_test_final, test_df = preprocess_data(train_df, test_df)
    best_model, X_train, X_val, y_train, y_val, results_df, models = train_and_evaluate_models(X, y)
    create_visualizations(X_train, y_train, best_model, X_val, y_val, models, train_df)
    save_model_and_predictions(best_model, X_test_final, test_df)
    print("\n✅ Task 1 Completed. All visualizations and predictions saved in 'visualizations/' and current folder.")

if __name__ == "__main__":
    main()
