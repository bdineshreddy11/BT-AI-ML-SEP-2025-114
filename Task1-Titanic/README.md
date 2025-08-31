Task 1: Titanic Survival Prediction
📊 Overview
This project predicts passenger survival on the Titanic using machine learning classification algorithms, demonstrating comprehensive data cleaning, exploratory analysis, feature engineering, and model building.

📁 Dataset
Source: Kaggle Titanic Competition

Training set: 891 passengers

Test set: 418 passengers

Target variable: Survival (0 = No, 1 = Yes)

🛠️ Implementation
1. Data Preprocessing & Cleaning
Handled Missing Values:

✅ Age (177 missing): Filled with median age by passenger title

✅ Embarked (2 missing): Filled with mode value

✅ Fare: Filled with median value

✅ Cabin (687 missing): Column dropped due to high missingness

Feature Engineering:

✅ Title Extraction: Created from Names (Mr, Mrs, Miss, Master, Rare)

✅ Age Grouping: Created categories (Child, Teen, Adult, Elderly)

✅ Family Features: Created FamilySize and IsAlone features

✅ Fare Grouping: Created FareGroup using quantile-based grouping

Data Transformation:

✅ Converted categorical features to numerical values

✅ Created dummy variables for multi-category features

✅ Dropped irrelevant columns (PassengerId, Name, Ticket, Cabin)

2. Exploratory Data Analysis (EDA)
✅ Survival analysis by Gender and Passenger Class

✅ Age and Fare distribution analysis

✅ Correlation analysis between features

✅ Comprehensive visualization of relationships

3. Model Implementation
✅ Logistic Regression with feature scaling

✅ Decision Tree Classifier with parameter optimization

✅ Hyperparameter Tuning using GridSearchCV

4. Model Evaluation
✅ Accuracy, Precision, Recall, F1-Score

✅ Confusion Matrices

✅ ROC Curve and AUC scores

✅ Cross-validation with 5 folds

📊 Exploratory Data Analysis
https://titanic_eda.png
Figure 1: Survival analysis by gender, passenger class, age, and fare distribution

📈 Model Performance
Confusion Matrices
https://confusion_matrix.png
Figure 2: Confusion matrices for Logistic Regression (left) and Decision Tree (right)

ROC Curve Analysis
https://roc_curve.png
Figure 3: ROC curves showing model performance across different thresholds

📋 Model Performance Results
Model	Accuracy	Precision	Recall	F1-Score	AUC
Logistic Regression	81.01%	0.81	0.81	0.81	0.8927
Decision Tree	82.12%	0.82	0.82	0.82	0.8863
🔍 Detailed Classification Reports
Logistic Regression
text
              precision    recall  f1-score   support

           0       0.83      0.85      0.84       105
           1       0.78      0.76      0.77        74

    accuracy                           0.81       179
   macro avg       0.80      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179
Decision Tree (Optimized)
text
              precision    recall  f1-score   support

           0       0.83      0.88      0.85       105
           1       0.81      0.74      0.77        74

    accuracy                           0.82       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179
🎯 Hyperparameter Tuning Results
Best Parameters for Decision Tree:

max_depth: 3

min_samples_leaf: 1

min_samples_split: 2

Best Cross-Validation Accuracy: 81.74%

📊 Cross-Validation Performance
Model	Mean Accuracy	Standard Deviation
Logistic Regression	82.02%	± 3.82%
Decision Tree	81.74%	± 5.16%
🔝 Top 10 Feature Importance
Feature	Importance Score
Title_Mr	62.20%
Pclass	20.41%
FamilySize	9.69%
Title_Rare	7.05%
FareGroup_3	0.41%
Parch	0.24%
SibSp	0.00%
Embarked	0.00%
IsAlone	0.00%
Title_Miss	0.00%
💡 Key Insights & Findings
Optimal Decision Tree Depth: Maximum depth of 3 provided best performance

Top Predictive Feature: "Title_Mr" (being addressed as Mr.) was the most important predictor

Class Impact: Passenger class (Pclass) was the second most important feature

Family Size: FamilySize emerged as the third most significant predictor

Model Consistency: Both models showed strong and consistent performance across validation methods

Women had significantly higher survival rates than men

Higher-class passengers had better survival chances

Children and elderly had better survival rates

Family size impacted survival probability

Decision Tree outperformed Logistic Regression

🚀 Extensions Implemented
✅ Hyperparameter Tuning - Optimized Decision Tree using GridSearchCV

✅ ROC Curve & AUC Analysis - Advanced model evaluation metrics

✅ Cross-Validation - 5-fold CV for reliable performance estimates

✅ Model Persistence - Saved best model using joblib

✅ Feature Importance - Identified most predictive features

📂 Project Structure
text
Task1-Titanic/
├── data/
│   ├── train.csv
│   └── test.csv
├── titanic_analysis.py
├── titanic_analysis.ipynb
├── best_decision_tree_model.pkl
├── titanic_eda.png
├── confusion_matrix.png
├── roc_curve.png
├── requirements.txt
└── README.md
🚀 Installation & Usage
bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the analysis
python titanic_analysis.py

# Or open Jupyter notebook
jupyter notebook titanic_analysis.ipynb
📋 Requirements
Python 3.7+

pandas

numpy

matplotlib

seaborn

scikit-learn

jupyter (for notebook version)

⚠️ Note on Warnings
The code includes some pandas future warnings related to chained assignment and in-place operations. These are cosmetic warnings that don't affect functionality and will be addressed in future pandas versions. The models perform optimally despite these warnings.