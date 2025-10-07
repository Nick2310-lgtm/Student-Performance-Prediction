"""
evaluate.py
Plot and compare model performance from metrics_summary.csv
"""
from utils import RESULTS_DIR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if not os.path.exists(csv):
        print("metrics_summary.csv not found. Run train_models.py first.")
        return

    df = pd.read_csv(csv)
    df.set_index("model")[["accuracy", "precision", "recall", "f1", "kappa"]].plot.bar(rot=0, figsize=(8, 4))
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
