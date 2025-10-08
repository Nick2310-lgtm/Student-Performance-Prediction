from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / "results" / "metrics" / "results.json"

def show_results(path=METRICS_PATH):
    if not path.exists():
        print(f"No metrics found at {path}. Run train_models.py first.")
        return

    with open(path) as f:
        results = json.load(f)

    print("\nModel Performance Summary")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:15} | Accuracy: {metrics['accuracy']:.4f} | F1-score: {metrics['f1_score']:.4f}")
    print("-" * 45)

if __name__ == "__main__":
    show_results()
