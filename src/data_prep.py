"""
data_prep.py
Preprocess Kaggle dataset and save preprocessed.csv
"""
from utils import load_raw, preprocess, DATA_DIR
import os

def main():
    df = load_raw()
    df = preprocess(df)
    out = os.path.join(DATA_DIR, "preprocessed.csv")
    df.to_csv(out, index=False)
    print(f"Preprocessed dataset saved to {out}")

if __name__ == "__main__":
    main()
