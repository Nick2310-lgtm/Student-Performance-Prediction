"""
utils.py
Common helper functions for all modules.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_raw():
    """Load raw Kaggle UCI CSV(s)."""
    files = ["student-mat.csv", "student-por.csv"]
    dfs = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, sep=';'))
    if not dfs:
        raise FileNotFoundError("No dataset files found. Place student-mat.csv or student-por.csv in /data/")
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    """Cleans, encodes, scales, and creates target."""
    df = df.dropna().copy()
    df["final_result"] = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail")
    cat_cols = df.select_dtypes(include="object").columns.drop("final_result", errors="ignore")
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def load_preprocessed():
    """Loads preprocessed.csv if exists, else processes raw."""
    path = os.path.join(DATA_DIR, "preprocessed.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    df = preprocess(load_raw())
    df.to_csv(path, index=False)
    return df

def save_csv(df, name):
    """Save any dataframe to /results/ folder."""
    out = os.path.join(RESULTS_DIR, name)
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")
