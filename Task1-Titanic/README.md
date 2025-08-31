Task 1: Titanic Survival Prediction
 Overview
This project predicts passenger survival on the Titanic using machine learning classification algorithms, demonstrating data cleaning, exploratory analysis, and model building.

 Dataset
Source: Kaggle Titanic Competition

Training set: 891 passengers

Test set: 418 passengers

Target variable: Survival (0 = No, 1 = Yes)

Implementation
1. Data Preprocessing & Cleaning
Handled Missing Values:

Age: Filled with median age by passenger title

Embarked: Filled with mode

Fare: Filled with median

Cabin: Dropped due to high missingness

Feature Engineering:

Extracted Title from Names (Mr, Mrs, Miss, Master, Rare)

Created AgeGroup (Child, Teen, Adult, Elderly)

Created FamilySize and IsAlone features

Created FareGroup using quantile-based grouping

Data Transformation:

Converted categorical features to numerical

Created dummy variables for multi-category features

Dropped irrelevant columns

2. Exploratory Data Analysis (EDA)
Survival analysis by Gender and Passenger Class

Age and Fare distribution analysis

Correlation analysis between features

3. Model Implementation
Logistic Regression with feature scaling

Decision Tree Classifier with parameter optimization

Hyperparameter Tuning using GridSearchCV

 Model Evaluation
Accuracy, Precision, Recall, F1-Score

Confusion Matrices

ROC Curve and AUC scores

Cross-validation with 5 folds

Results
Model Performance:
Model	Accuracy	Precision	Recall	F1-Score	AUC
Logistic Regression	79.33%	0.79	0.79	0.79	0.8575
Decision Tree	81.56%	0.82	0.82	0.82	0.8190
Key Insights:
Women had significantly higher survival rates

Higher-class passengers had better survival chances

Children and elderly had better survival rates

Family size impacted survival probability

Decision Tree outperformed Logistic Regression

Extensions Implemented
Hyperparameter Tuning - Optimized Decision Tree parameters

ROC Curve & AUC Analysis - Advanced model evaluation

Cross-Validation - 5-fold CV for reliable performance estimates

Model Persistence - Saved best model using joblib

Feature Importance - Identified most predictive features

Project Structure
text
Task1-Titanic/
├── data/
│   ├── train.csv
│   └── test.csv
├── titanic_analysis.py
├── best_decision_tree_model.pkl
├── titanic_eda.png
├── confusion_matrix.png
├── roc_curve.png
└── README.md
Installation & Usage
bash
# Install requirements
pip install pandas numpy matplotlib seaborn scikit-learn

# Run analysis
python titanic_analysis.py
Requirements
Python 3.7+

pandas, numpy, matplotlib, seaborn, scikit-learn

 Author
Name: [Your Name]

Registration Number: [Your Registration Number]

Repository: BT-AI-ML-SEP-2025-114