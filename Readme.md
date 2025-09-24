# End-to-End ML Pipeline for Customer Churn Prediction

# Objective of the Task
The goal of this task is to build a reusable and production-ready machine learning pipeline to predict customer churn for a telecommunications company. The pipeline should handle preprocessing, model training, hyperparameter tuning, evaluation, and enable easy deployment.

# Methodology / Approach
1. Dataset Loading & Preprocessing
   - Loaded the Telco Customer Churn dataset.
   - Handled missing values using imputation.
   - Removed duplicate rows.
   - Separated features into numeric and categorical.
   - Built a preprocessing pipeline using `ColumnTransformer`:
     - Numeric: `SimpleImputer` (median) + `StandardScaler`
     - Categorical: `SimpleImputer` (most frequent) + `OneHotEncoder`
   - Split data into training and test sets (80%-20%).

2. Model Development & Training
   - Built pipelines combining preprocessing and models:
     - Logistic Regression
     - Random Forest Classifier
   - Trained baseline models without hyperparameter tuning.
   - Applied `GridSearchCV` for hyperparameter tuning:
     - Logistic Regression with 10-fold CV
     - Random Forest with 5-fold CV

3. Evaluation
   - Metrics used: Accuracy, ROC AUC, Precision, Recall, F1-score.
   - Compared untuned vs. tuned models for both Logistic Regression and Random Forest.
   - Analyzed metrics for churned customers (target class).

4. Pipeline Export
   - Exported trained pipelines using `joblib` for reusability and production readiness.

# Key Results / Observations
- Preprocessing pipelines handle missing values, scaling, and encoding automatically.
- Hyperparameter tuning stabilizes or slightly improves model performance.
- Logistic Regression and Random Forest are both effective:
  - Random Forest achieved slightly higher accuracy (0.802)  
  - Logistic Regression had slightly better ROC AUC and recall for churned customers.
- Model selection depends on the priority: catching more churned customers (recall) vs. overall accuracy.
- The workflow ensures end-to-end ML pipeline construction, evaluation, tuning, and deployment readiness.
