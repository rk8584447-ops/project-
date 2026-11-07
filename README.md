# Explainable AI for Clinical Decision Support

Academic prototype for the project synopsis **Explainable AI-Based Machine Learning Framework for Clinical Decision Support Systems**. It compares four classifiers on the public UCI Heart Disease dataset and shows model metrics plus a local SHAP explanation for the selected test case.

> **Demonstration only.** This software is not a medical device and is not validated for diagnosis, screening, treatment, or patient care. Use only the public de-identified dataset. Do not enter real patient information.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads dataset 45 from the UCI Machine Learning Repository and caches it in `data/`. An internet connection is needed the first time. To train and save the experiment outside the interface:

```bash
python -m src.train
```

Training outputs are written to `artifacts/` and `reports/`. Metrics are computed from a stratified held-out test split; the app does not claim clinical validity.

## Dataset and target

The implementation uses the UCI Heart Disease repository dataset (303 Cleveland records, 13 selected predictors). The original outcome `num` is converted to the synopsis's binary target: `0` means no disease finding in this dataset; `1–4` are grouped as disease finding. Rows without a usable target are removed. Missing predictor values are imputed inside each model's training pipeline to avoid test-data leakage. Categorical variables are one-hot encoded; continuous variables are median-imputed and standardized for Logistic Regression.

This is a small, historical, single-source dataset. It is not representative of current clinical populations. Dataset labels and predicted probabilities must not be interpreted as an individual's clinical diagnosis or calibrated risk.

## Project layout

```text
app.py                 Streamlit dashboard
src/data.py            UCI loading and target preparation
src/modeling.py        Leakage-safe preprocessing, training, evaluation, SHAP
src/train.py           Reproducible training and report generation
data/                  Downloaded public dataset (created on first run)
artifacts/             Saved model bundle (created by training)
reports/               Metrics and model comparison (created by training)
```

## What the prototype shows

- Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting comparison.
- Stratified train/test split and stratified cross-validation on the training partition.
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve.
- Per-case model-agnostic SHAP feature contributions (when SHAP is installed and explanation is requested).
- Global permutation feature importance and an explicit limitations notice.

The default is a fixed seed (`42`) so the experiment is reproducible. With only 303 rows, the displayed scores have high uncertainty and should be treated as an educational result, not a performance guarantee.

## Team

- Zaid Hasan Khan (2300101438)
- Raza Khan (2300104164)
- Soyab Khan (2300101263)

Supervisor: Dr. S. H. Abbas Rizvi  
Integral University, Lucknow | Academic year 2026–2027
