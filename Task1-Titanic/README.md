##  <span style="color:#773344">Task 1: Titanic Survival Prediction</span>

📊 Overview

This project predicts passenger survival on the Titanic using machine learning classification algorithms, demonstrating comprehensive data cleaning, exploratory analysis, feature engineering, and model building.

📁 Dataset
Source: Kaggle Titanic Competition

Training set: 891 passengers

Test set: 418 passengers

Target variable: Survival (0 = No, 1 = Yes)

## 🛠️ <span style="color:#773344">Implementation</span>
1. Data Preprocessing & Cleaning
Handled Missing Values:

✅ Age (177 missing): Filled with median age by passenger title

✅ Embarked (2 missing): Filled with mode value

✅ Fare: Filled with median value

✅ Cabin (687 missing): Column dropped due to high missingness

 ##  <span style="color:#773344">Feature Engineering:</span>

✅ Title Extraction: Created from Names (Mr, Mrs, Miss, Master, Rare)

✅ Age Grouping: Created categories (Child, Teen, Adult, Elderly)

✅ Family Features: Created FamilySize and IsAlone features

✅ Fare Grouping: Created FareGroup using quantile-based grouping

##  <span style="color:#773344">Data Transformation: </span>

✅ Converted categorical features to numerical values

✅ Created dummy variables for multi-category features

✅ Dropped irrelevant columns (PassengerId, Name, Ticket, Cabin)

##  <span style="color:#773344">2. Exploratory Data Analysis (EDA)  </span>

✅ Survival analysis by Gender and Passenger Class

✅ Age and Fare distribution analysis

✅ Correlation analysis between features

✅ Comprehensive visualization of relationships

 ##  <span style="color:#773344">3. Model Implementation </span>
✅ Logistic Regression with feature scaling

✅ Decision Tree Classifier with parameter optimization

✅ Hyperparameter Tuning using GridSearchCV

##  <span style="color:#773344">4. Model Evaluation   </span>
✅ Accuracy, Precision, Recall, F1-Score

✅ Confusion Matrices

✅ ROC Curve and AUC scores

✅ Cross-validation with 5 folds

## 📊 <span style="color:#2E86AB">Exploratory Data Analysis</span>
![EDA Visualization](https://github.com/bdineshreddy11/BT-AI-ML-SEP-2025-114/raw/main/Task1-Titanic/titanic_eda.png)

Figure 1: Survival analysis by gender, passenger class, age, and fare distribution
## 📈 <span style="color:#A23B72">Model Performance</span>

Confusion Matrices:
![Confusion Matrix](https://github.com/bdineshreddy11/BT-AI-ML-SEP-2025-114/raw/main/Task1-Titanic/confusion_matrix.png)
Figure 2: Confusion matrices for Logistic Regression (left) and Decision Tree (right)

ROC Curve Analysis:
![ROC Curve](https://github.com/bdineshreddy11/BT-AI-ML-SEP-2025-114/raw/main/Task1-Titanic/roc_curve.png)
Figure 3: ROC curves showing model performance across different thresholds

## 📋 <span style="color:#F18F01">Model Performance Results</span>
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
##🎯  <span style="color:#773344">Hyperparameter Tuning Results</span>
Best Parameters for Decision Tree:

max_depth: 3

min_samples_leaf: 1

min_samples_split: 2

Best Cross-Validation Accuracy: 81.74%

## <span style="color:#FF6B6B">📊 Cross-Validation Performance </span>
Model	Mean Accuracy	Standard Deviation

| Model | <span style="color:#2E86AB">Accuracy</span> | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | <span style="color:green; font-weight:bold">81.01%</span> | 0.81 | 0.81 | 0.81 | 0.8927 |
| Decision Tree | <span style="color:green; font-weight:bold">82.12%</span> | 0.82 | 0.82 | 0.82 | 0.8863 |



## <span style="color:#FF6B6B">🔝 Top 10 Feature Importance  </span>
| Feature       | Importance Score |
|---------------|------------------|
| Title_Mr      | 62.20%           |
| Pclass        | 20.41%           |
| FamilySize    | 9.69%            |
| Title_Rare    | 7.05%            |
| FareGroup_3   | 0.41%            |
| Parch         | 0.24%            |
| SibSp         | 0.00%            |
| Embarked      | 0.00%            |
| IsAlone       | 0.00%            |
| Title_Miss    | 0.00%            |

 ## <span style="color:#FF6B6B">💡 Key Insights & Findings </span>
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

 ## 🚀 <span style="color:#FF6B6B">Extensions Implemented</span>
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

👨‍💻 Author
Name: Dinesh Reddy

Registration Number: 114

Repository: BT-AI-ML-SEP-2025-114
