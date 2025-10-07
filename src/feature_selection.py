"""
feature_selection.py
Select top features using RandomForestClassifier
"""
from utils import load_preprocessed, RESULTS_DIR, save_csv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

def main(top_k=10):
    df = load_preprocessed()
    X, y = df.drop(columns=["final_result"]), df["final_result"]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    save_csv(imp.reset_index(names=["feature", "importance"]), "feature_importance.csv")

    top_feats = imp.head(top_k).index.tolist()
    with open(os.path.join(RESULTS_DIR, "selected_features.txt"), "w") as f:
        f.write("\n".join(top_feats))

    joblib.dump(rf, os.path.join(RESULTS_DIR, "rf_selector.joblib"))
    print("Top features saved -> selected_features.txt")

if __name__ == "__main__":
    main()
