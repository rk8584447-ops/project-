"""Leakage-safe model training, metrics, and model-agnostic explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]


def make_pipeline(X: pd.DataFrame, model: Any, scale_numeric: bool = False) -> Pipeline:
    categorical = [name for name in CATEGORICAL if name in X.columns]
    numeric = [name for name in X.columns if name not in categorical]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", prep), ("classifier", model)])


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": make_pipeline(
            X, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE), True
        ),
        "Decision Tree": make_pipeline(
            X, DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": make_pipeline(
            X, RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        ),
        "Gradient Boosting": make_pipeline(
            X, GradientBoostingClassifier(n_estimators=100, learning_rate=0.04, max_depth=2, random_state=RANDOM_STATE)
        ),
    }


def train_experiment(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    models = build_models(X)
    results: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    cv_folds = min(5, int(y_train.value_counts().min()))
    for name, pipeline in models.items():
        cv = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="roc_auc", n_jobs=1)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "CV ROC-AUC (mean)": float(cv.mean()),
            "CV ROC-AUC (std)": float(cv.std()),
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1-score": float(f1_score(y_test, pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, proba)),
        })
        fitted[name] = pipeline

    # Select by CV ROC-AUC, avoiding selection on the held-out test scores.
    best_name = max(results, key=lambda row: row["CV ROC-AUC (mean)"])["Model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    best_proba = best_model.predict_proba(X_test)[:, 1]
    return {
        "models": fitted,
        "best_name": best_name,
        "best_model": best_model,
        "results": pd.DataFrame(results).sort_values("CV ROC-AUC (mean)", ascending=False).reset_index(drop=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": best_pred,
        "probabilities": best_proba,
        "confusion_matrix": confusion_matrix(y_test, best_pred, labels=[0, 1]),
        "cv_folds": cv_folds,
    }


def explain_case(experiment: dict[str, Any], row: pd.DataFrame, max_evals: int = 200) -> pd.DataFrame:
    """Return SHAP values for disease-finding probability for a single input row."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install project requirements and restart the app.") from exc

    X_train = experiment["X_train"]
    # Small representative background keeps the model-agnostic KernelExplainer usable.
    background = X_train.sample(n=min(25, len(X_train)), random_state=RANDOM_STATE)
    model = experiment["best_model"]
    def predict_probability(values: Any) -> np.ndarray:
        # KernelExplainer passes NumPy arrays, while this pipeline selects its
        # columns by name. Restore the dataframe schema before inference.
        records = pd.DataFrame(values, columns=X_train.columns)
        return model.predict_proba(records)

    explainer = shap.KernelExplainer(predict_probability, background)
    values = explainer.shap_values(row, nsamples=max_evals)
    if isinstance(values, list):
        contribution = np.asarray(values[1]).reshape(-1)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            contribution = arr[0, :, 1].reshape(-1)
        else:
            contribution = arr.reshape(-1)
    return pd.DataFrame({
        "Feature": row.columns,
        "Input value": [row.iloc[0][column] for column in row.columns],
        "SHAP value": contribution,
    }).assign(Abs=lambda data: data["SHAP value"].abs()).sort_values("Abs", ascending=False).drop(columns="Abs").reset_index(drop=True)


def global_permutation_importance(experiment: dict[str, Any]) -> pd.DataFrame:
    """Compute held-out permutation importance in original input feature space."""
    from sklearn.inspection import permutation_importance

    output = permutation_importance(
        experiment["best_model"], experiment["X_test"], experiment["y_test"],
        scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
    )
    return pd.DataFrame({
        "Feature": experiment["X_test"].columns,
        "Importance (mean ROC-AUC decrease)": output.importances_mean,
        "Std": output.importances_std,
    }).sort_values("Importance (mean ROC-AUC decrease)", ascending=False).reset_index(drop=True)
