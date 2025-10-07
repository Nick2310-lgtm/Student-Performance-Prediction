# TYBTechML_Group55_StudentPerformancePrediction

**Based on:** Ahmed, E. (2024) *Student Performance Prediction Using Machine Learning Algorithms*  
**Dataset:** [Kaggle - UCI Student Performance Dataset](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)

---

## ⚙️ Run Instructions
```bash
pip install -r requirements.txt
python src/data_prep.py
python src/feature_selection.py
python src/train_models.py
python src/evaluate.py
```
All results (accuracy, F1, kappa, etc.) appear in `/results/metrics_summary.csv`.

---

## 📁 Folder Summary
| Folder | Contents |
|--------|-----------|
| `data/` | dataset CSVs + README |
| `src/` | all Python scripts |
| `results/` | auto-generated results |
| `docs/` | methodology & extension notes |
| `Paper_PDF/` | Ahmed (2024) paper |
