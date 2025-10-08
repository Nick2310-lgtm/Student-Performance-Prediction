# Methodology

This document explains how the original paper’s methodology was reproduced and adapted to the new dataset.

## Data Preprocessing
- Categorical features are encoded using one-hot encoding.
- Numerical features are scaled using standardization.
- Missing values (if any) are handled appropriately to match the workflow of the original study.

## Model Training
- Implemented the same models as the original paper:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Naive Bayes
- Hyperparameters are slightly adjusted to suit the Kaggle student performance dataset.

## Evaluation
- Performance metrics such as Accuracy and F1-score are calculated for each model.
- Evaluation results are saved to `results/metrics/results.json`.
- Trained models and preprocessing objects are saved in `results/models/` for reproducibility.

This methodology ensures faithful reproduction of the original study while adapting it to a new dataset.
