from pathlib import Path
import os, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocess import prepare_data

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "student-mat.csv"
RESULTS_MODELS = REPO_ROOT / "results" / "models"
RESULTS_METRICS = REPO_ROOT / "results" / "metrics"

RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
RESULTS_METRICS.mkdir(parents=True, exist_ok=True)

def train_all():
    X_train, X_test, y_train, y_test, preproc = prepare_data(DATA_PATH)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4)
        }
        joblib.dump(model, RESULTS_MODELS / f"{name}.joblib")

    with open(RESULTS_METRICS / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(preproc, RESULTS_MODELS / "preprocessor.joblib")

if __name__ == "__main__":
    train_all()
