# Methodology (Ahmed 2024 Reproduction)

**Goal:** Predict student performance (Pass/Fail) using machine learning models.

## Workflow
1. **Data Preprocessing**
   - Combine `student-mat.csv` and `student-por.csv`.
   - Handle missing values, label encode categorical columns.
   - Scale numeric columns and create `final_result` = Pass/Fail.

2. **Feature Selection**
   - RandomForest feature importances used to pick top attributes.

3. **Model Training**
   - Algorithms: SVM, Decision Tree, KNN, Naive Bayes.
   - 10-fold Stratified Cross Validation.
   - Hyperparameter tuning using GridSearchCV.

4. **Evaluation**
   - Accuracy, Precision, Recall, F1, Cohen’s Kappa.
   - Compare results visually using bar plots.

This matches Ahmed (2024), *Student Performance Prediction Using Machine Learning Algorithms*,  
except the dataset is replaced with the open-source **UCI Kaggle dataset**.
