# TYBTechML_Group55_StudentPerformancePrediction

**Reproducing:** Ahmed, E. (2024) "Student Performance Prediction Using Machine Learning Algorithms"  
**Extension dataset:** UCI Student Performance (Portuguese schools) — Kaggle: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set

## Summary
This repository reproduces the Ahmed (2024) pipeline (preprocessing → feature selection → clustering → modeling → evaluation) and applies it to the UCI Student dataset.

## Minimal run instructions
1. Place the dataset CSV(s) in `data/` (see `data/README_DATA.txt`):
   - `student-mat.csv` and/or `student-por.csv` (Kaggle).  
2. Create and activate Python environment:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Run the pipeline:
```bash
python src/data_prep.py
python src/feature_selection.py
python src/train_models.py
python src/evaluate.py
```
4. Results (metrics, models, plots) will be in `results/`.

## Files included
- `src/` — core Python scripts
- `data/README_DATA.txt` — dataset link and notes
- `docs/` — methodology & extension explanations
- `Paper_PDF/` — include `TYBTechML_Group1_StudentPerformancePrediction.pdf` (paper)
- `requirements.txt`, `.gitignore`

## Notes
- The code creates `results/` at runtime and writes intermediate outputs there.
- Default target: binary `Pass` if `G3 >= 10`, else `Fail`. See `src/data_prep.py` to change to multiclass.
