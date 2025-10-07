"""
train_models.py
Train SVM, DecisionTree, KNN, and NaiveBayes models on preprocessed dataset.
"""
from utils import load_preprocessed, RESULTS_DIR, save_csv
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import joblib
import os

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred)
    }

def main():
    df = load_preprocessed()
    X, y = df.drop(columns=["final_result"]), df["final_result"]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "SVM": (Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
                {"clf__C": [1, 10], "clf__kernel": ["linear", "rbf"]}),
        "DecisionTree": (DecisionTreeClassifier(random_state=42),
                {"max_depth": [3, 5, 7]}),
        "KNN": (Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
                {"clf__n_neighbors": [3, 5, 8]}),
        "NaiveBayes": (Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]), {})
    }

    results = []
    for name, (model, params) in models.items():
        print(f"Training {name}...")
        if params:
            gcv = GridSearchCV(model, params, cv=skf, scoring="accuracy", n_jobs=-1)
            gcv.fit(X, y)
            best = gcv.best_estimator_
        else:
            model.fit(X, y)
            best = model

        y_pred = cross_val_predict(best, X, y, cv=skf)
        m = evaluate(y, y_pred)
        results.append({"model": name, **m})
        joblib.dump(best, os.path.join(RESULTS_DIR, f"{name}_model.joblib"))
        print(f"{name} done -> Accuracy={m['accuracy']:.3f}, Kappa={m['kappa']:.3f}")

    save_csv(pd.DataFrame(results), "metrics_summary.csv")
    print("All results saved to results/metrics_summary.csv")

if __name__ == "__main__":
    main()
