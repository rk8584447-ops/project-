"""Train the model comparison and save reproducible artifacts/reports."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay

from src.data import dataset_signature, load_heart_data
from src.modeling import explain_case, global_permutation_importance, train_experiment

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
        "dataset_signature": dataset_signature(X, y),
    }
    joblib.dump(bundle, ROOT / "artifacts" / "heart_disease_model.joblib")
    joblib.dump(
        {
            "experiment": experiment,
            "dataset_signature": dataset_signature(X, y),
            "feature_names": list(X.columns),
            "random_state": 42,
        },
        ROOT / "artifacts" / "project_experiment.joblib",
        compress=3,
    )
    experiment["results"].to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    # Save figure/report evidence so the submitted results can be reproduced.
    cm = experiment["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Class 0", "Class 1"],
           yticklabels=["Class 0", "Class 1"], xlabel="Predicted label", ylabel="Actual label",
           title=f"Confusion matrix: {experiment['best_name']}")
    for (row, column), value in np.ndenumerate(cm):
        ax.text(column, row, str(value), ha="center", va="center", color="#17324d", fontsize=14)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    RocCurveDisplay.from_predictions(
        experiment["y_test"], experiment["probabilities"], ax=ax, name=experiment["best_name"]
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7a8793", linewidth=1)
    ax.set_title("ROC curve on held-out test split")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    importance = global_permutation_importance(experiment)
    importance.to_csv(ROOT / "reports" / "global_permutation_importance.csv", index=False)
    shown_importance = importance.head(10).sort_values("Importance (mean ROC-AUC decrease)")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(shown_importance["Feature"], shown_importance["Importance (mean ROC-AUC decrease)"], color="#397c8e")
    ax.set(xlabel="Mean decrease in held-out ROC-AUC", title="Global permutation importance")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "global_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sample = experiment["X_test"].iloc[[0]]
    case_explanation = explain_case(experiment, sample, max_evals=100)
    case_explanation.to_csv(ROOT / "reports" / "sample_shap_explanation.csv", index=False)
    shown_explanation = case_explanation.head(8).sort_values("SHAP value")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#bb4b45" if value > 0 else "#367a8b" for value in shown_explanation["SHAP value"]]
    ax.barh(shown_explanation["Feature"], shown_explanation["SHAP value"], color=colors)
    ax.axvline(0, color="#555", linewidth=.8)
    ax.set(xlabel="SHAP contribution to class 1 output", title="Local explanation for held-out case")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "sample_shap_explanation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "records": len(X),
        "features": X.shape[1],
        "test_records": len(experiment["y_test"]),
        "cross_validation_folds": experiment["cv_folds"],
        "selected_model": experiment["best_name"],
        "confusion_matrix_labels_0_1": cm.tolist(),
        "false_positive_count": int(cm[0, 1]),
        "false_negative_count": int(cm[1, 0]),
        "sample_explanation_index": int(sample.index[0]),
        "sample_explanation_prediction": int(experiment["best_model"].predict(sample)[0]),
        "sample_explanation_probability_class_1": float(experiment["best_model"].predict_proba(sample)[0, 1]),
    }
    pd.Series(summary, name="value").to_json(ROOT / "reports" / "run_summary.json", indent=2)
    print(f"Rows: {len(X)} | Features: {X.shape[1]} | CV folds: {experiment['cv_folds']}")
    print(f"Selected using training-partition CV ROC-AUC: {experiment['best_name']}")
    print(experiment["results"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Saved model, metrics, confusion matrix, ROC curve, global importance, SHAP case explanation, and run summary.")


if __name__ == "__main__":
    main()
