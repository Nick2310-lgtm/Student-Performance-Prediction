# TYBTechML_Group55_StudentPerformance_UCI

**Title:** Student Performance Prediction Using Machine Learning Algorithms  
**Based on:** Ahmed, E. (2024), *Applied Computational Intelligence and Soft Computing*  
**Dataset:** UCI Student Performance (Portuguese schools) — [Kaggle link](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)

## 1️⃣ Overview
This project reproduces the ML workflow of Ahmed (2024) using an open Kaggle dataset instead of the original university data.
The pipeline includes:
- Data preprocessing and feature encoding
- Random Forest–based feature selection
- K-Means clustering
- Model training with SVM, Decision Tree, KNN, and Naive Bayes
- Hyperparameter tuning and evaluation (Accuracy, Precision, Recall, Cohen’s Kappa)

## 2️⃣ Dataset
- Files: `student-mat.csv` and `student-por.csv`
- Total records: ~650 students
- Features: demographics, academic grades (G1, G2, G3), study time, absences, family info
- Target: derived `final_result` column (Pass/Fail)

## 3️⃣ How to Run
```bash
pip install -r requirements.txt
python src/data_prep.py
python src/feature_selection.py
python src/clustering.py
python src/train_models.py
