"""Train the model comparison and save reproducible artifacts/reports."""

from pathlib import Path

import joblib

from src.data import load_heart_data
from src.modeling import train_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    X, y = load_heart_data()
    experiment = train_experiment(X, y)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    bundle = {
        "best_name": experiment["best_name"],
        "best_model": experiment["best_model"],
        "feature_names": list(X.columns),
        "target_definition": "UCI num: 0 -> 0; 1-4 -> 1",
        "random_state": 42,
    }
    joblib.dump(bundle, ROOT / "artifacts" / "heart_disease_model.joblib")
    experiment["results"].to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    print(f"Rows: {len(X)} | Features: {X.shape[1]} | CV folds: {experiment['cv_folds']}")
    print(f"Selected using training-partition CV ROC-AUC: {experiment['best_name']}")
    print(experiment["results"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Saved artifacts/heart_disease_model.joblib and reports/model_comparison.csv")


if __name__ == "__main__":
    main()
