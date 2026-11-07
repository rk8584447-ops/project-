"""Streamlit interface for the explainable heart-disease academic prototype."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import RocCurveDisplay

from src.data import dataset_signature, feature_metadata, load_heart_data
from src.modeling import explain_case, global_permutation_importance, train_experiment

st.set_page_config(page_title="Explainable CDSS | Heart Disease", page_icon="🫀", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .stApp {background:linear-gradient(180deg,#f2f8f7 0%,#f7f9fc 34rem,#f7f9fc 100%);}
    .main .block-container {max-width:1320px; padding:2.3rem 2rem 3rem;}
    [data-testid="stHeader"] {background:transparent;}
    .hero {position:relative; overflow:hidden; padding:2.1rem 2.3rem; margin:0 0 1.25rem; border-radius:1.4rem; color:white;
      background:linear-gradient(115deg,#103c48 0%,#126b70 58%,#37a18c 100%); box-shadow:0 16px 42px #16465324;}
    .hero:after {content:""; position:absolute; width:19rem; height:19rem; right:-5rem; top:-11rem; border:1px solid #ffffff35; border-radius:50%; box-shadow:0 0 0 2.4rem #ffffff0a,0 0 0 5rem #ffffff08;}
    .hero-kicker {font-size:.75rem; font-weight:750; letter-spacing:.15em; text-transform:uppercase; color:#bce9dc;}
    .hero h1 {font-size:clamp(2rem,4vw,3.05rem); line-height:1.08; margin:.7rem 0 .65rem; color:white; letter-spacing:-.04em;}
    .hero p {max-width:48rem; margin:0; color:#e1f2ef; font-size:1.04rem;}
    .hero-tag {display:inline-block; margin-top:1.25rem; padding:.35rem .7rem; border:1px solid #ffffff48; border-radius:99px; color:#e7fffa; font-size:.8rem;}
    .risk-note {padding:.95rem 1.1rem; margin:.7rem 0 1.45rem; background:#fff7e9; border:1px solid #f3ddb0; border-left:5px solid #db9b2b; border-radius:.75rem; color:#624718; font-size:.93rem;}
    div[data-testid="stMetric"] {background:white; padding:1rem 1.1rem; border:1px solid #e5ecec; border-radius:1rem; box-shadow:0 5px 18px #173d4c09;}
    div[data-testid="stMetricLabel"] {color:#607982; font-size:.83rem;}
    div[data-testid="stMetricValue"] {color:#123d49; font-weight:720;}
    .stTabs [data-baseweb="tab-list"] {gap:.45rem; border-bottom:1px solid #dfe8e8;}
    .stTabs [data-baseweb="tab"] {height:3.2rem; padding:0 1rem; border-radius:.7rem .7rem 0 0; color:#587078; font-weight:650;}
    .stTabs [aria-selected="true"] {background:#e4f2ef; color:#116b69;}
    .section-intro {padding:1rem 1.1rem; margin:.3rem 0 1rem; background:#fff; border:1px solid #e5ecec; border-radius:.85rem; color:#49616a;}
    .section-intro strong {color:#173f49;}
    h2,h3 {color:#153f49; letter-spacing:-.02em;}
    div[data-testid="stForm"] {padding:1.2rem; background:white; border:1px solid #e4ebeb; border-radius:1rem; box-shadow:0 5px 18px #173d4c08;}
    .stButton button, .stFormSubmitButton button {border-radius:.7rem; font-weight:650; padding:.55rem 1rem;}
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {background:linear-gradient(110deg,#126b70,#239884); border:0;}
    div[data-testid="stDataFrame"] {border:1px solid #e2eaea; border-radius:.8rem; overflow:hidden;}
    div[data-testid="stAlert"] {border-radius:.8rem;}
    .footer {padding-top:.6rem; color:#73868d; font-size:.82rem;}
    @media (max-width:700px) {.main .block-container {padding:1.2rem 1rem 2rem;} .hero {padding:1.55rem 1.3rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<section class="hero"><div class="hero-kicker">PCS 26 / 125 · Explainable AI</div>'
    '<h1>Clinical Decision Support<br>Model Explorer</h1>'
    '<p>Compare machine learning models, explore a sample input, and see which features influenced the model output.</p>'
    '<span class="hero-tag">🫀 &nbsp;UCI Heart Disease · Academic prototype</span></section>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="risk-note"><strong>For academic demonstration only.</strong> This prototype is not clinically validated and must not be used for diagnosis, treatment, screening, or patient care. Do not enter real patient information. A model explanation does not establish that a prediction is correct.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the bundled dataset and model comparison…")
def get_experiment():
    X, y = load_heart_data()
    artifact_path = Path(__file__).resolve().parent / "artifacts" / "project_experiment.joblib"
    if artifact_path.exists():
        try:
            bundle = joblib.load(artifact_path)
            if (
                bundle.get("dataset_signature") == dataset_signature(X, y)
                and bundle.get("feature_names") == list(X.columns)
                and bundle.get("random_state") == 42
            ):
                return X, y, bundle["experiment"], "Saved project run"
        except Exception:
            pass
    return X, y, train_experiment(X, y), "Fresh run"


try:
    X, y, experiment, run_source = get_experiment()
except Exception as exc:
    st.error(f"The project could not prepare the dataset: {exc}")
    st.info("Install the project dependencies from requirements.txt. The bundled dataset is available in data/heart_disease.csv.")
    st.stop()

if run_source == "Fresh run":
    st.info("No matching saved experiment was found, so the app trained a fresh model comparison. Run `python -m src.train` to refresh the saved project bundle.")

best_row = experiment["results"].iloc[0]
metric_cols = st.columns(4)
metric_cols[0].metric("Dataset records", f"{len(X):,}", help="Public Cleveland subset included with the project")
metric_cols[1].metric("Input features", f"{X.shape[1]}", help="Selected structured predictors")
metric_cols[2].metric("Selected model", experiment["best_name"])
metric_cols[3].metric("Held-out ROC-AUC", f"{best_row['ROC-AUC']:.3f}", help="Single stratified test split; not clinical validation")

tab_overview, tab_predict, tab_method = st.tabs(["📊  Model overview", "🧪  Explore a case", "📘  Method & safety"])

with tab_overview:
    st.subheader("Model performance")
    st.markdown(
        f'<div class="section-intro"><strong>{experiment["best_name"]}</strong> was selected using mean ROC-AUC from '
        f'{experiment["cv_folds"]}-fold cross-validation on training data. Scores below use the separate '
        f'{len(experiment["y_test"])}-record held-out split. <span style="color:#71858a">{run_source} · seed 42</span></div>', unsafe_allow_html=True)
    st.dataframe(
        experiment["results"].style.format({column: "{:.3f}" for column in experiment["results"].columns if column != "Model"})
            .background_gradient(subset=["CV ROC-AUC (mean)", "ROC-AUC"], cmap="GnBu", vmin=0, vmax=1),
        width="stretch",
        hide_index=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Where the model was right and wrong")
        cm = experiment["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.imshow(cm, cmap="GnBu")
        ax.set_xticks([0, 1], ["Class 0", "Class 1"])
        ax.set_yticks([0, 1], ["Class 0", "Class 1"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title("Held-out confusion matrix", pad=12)
        midpoint = max(cm.max(), 1) / 2
        for (row_idx, col_idx), value in np.ndenumerate(cm):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=16,
                    color="white" if value > midpoint else "#183c46", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="Confusion matrix showing actual and predicted dataset classes")
        plt.close(fig)
        st.caption("Target 0/1 reflects the UCI dataset grouping (num=0 vs num=1–4), not a clinical determination.")
    with right:
        st.markdown("**Selected model: ROC curve**")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        RocCurveDisplay.from_predictions(experiment["y_test"], experiment["probabilities"], ax=ax, name=experiment["best_name"])
        ax.plot([0, 1], [0, 1], linestyle="--", color="#8b949e", linewidth=1)
        ax.set_title("Held-out ROC curve", color="#17434c", pad=12)
        ax.set_facecolor("#fbfdfd")
        ax.grid(alpha=.18)
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="ROC curve for the selected model on the held-out test split")
        plt.close(fig)
    st.markdown("#### What features mattered overall?")
    st.caption("Permutation importance shows how held-out ROC-AUC changes when one feature is shuffled. It describes this model, not cause and effect.")
    if st.button("Calculate global importance", key="global_importance"):
        with st.spinner("Calculating permutation importance…"):
            importance = global_permutation_importance(experiment)
        top = importance.head(10).sort_values("Importance (mean ROC-AUC decrease)")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(top["Feature"], top["Importance (mean ROC-AUC decrease)"], color="#248b83")
        ax.set_xlabel("Held-out ROC-AUC decrease after shuffling")
        ax.set_title("Global permutation importance", color="#17434c", pad=12)
        ax.set_facecolor("#fbfdfd")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="Top global permutation importance features")
        plt.close(fig)
        st.dataframe(importance.style.format({"Importance (mean ROC-AUC decrease)": "{:.4f}", "Std": "{:.4f}"}), width="stretch", hide_index=True)

with tab_predict:
    st.subheader("Explore the model with a sample input")
    st.markdown('<div class="section-intro">Choose values using the dataset codes. Submit the form to see the model output and a local feature explanation. <strong>Use demonstration values only.</strong></div>', unsafe_allow_html=True)
    st.caption("Category options include plain-language labels from the [UCI dataset documentation](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease).")
    metadata = feature_metadata()
    with st.form("case_form"):
        entered = {}
        columns = st.columns(3)
        for index, feature in enumerate(X.columns):
            meta = metadata.get(feature, {"label": feature, "min": float(X[feature].min()), "max": float(X[feature].max()), "default": float(X[feature].median()), "step": 1.0})
            with columns[index % 3]:
                if "options" in meta:
                    options = meta["options"]
                    option_labels = meta.get("option_labels", {})
                    entered[feature] = st.selectbox(
                        meta["label"], options, index=options.index(meta["default"]),
                        format_func=lambda value, labels=option_labels: f"{value} · {labels.get(int(value), 'Dataset code')}",
                        help=meta.get("help"),
                    )
                else:
                    entered[feature] = st.number_input(
                        f"{meta['label']} ({meta.get('unit', '')})".strip(),
                        min_value=float(meta["min"]), max_value=float(meta["max"]),
                        value=float(meta["default"]), step=float(meta["step"]),
                        help=meta.get("help"),
                    )
        submitted = st.form_submit_button("Show model output  →", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame([{feature: entered[feature] for feature in X.columns}], columns=X.columns)
        probability = float(experiment["best_model"].predict_proba(row)[0, 1])
        predicted_class = int(experiment["best_model"].predict(row)[0])
        first, second, third = st.columns(3)
        first.metric("Predicted dataset class", f"Class {predicted_class}")
        second.metric("Model output for class 1", f"{probability:.1%}")
        third.metric("Selected model", experiment["best_name"])
        st.caption("Probability is the classifier's output on this academic dataset; it is not a calibrated personal risk estimate.")
        with st.spinner("Calculating local SHAP contributions… this may take a short while."):
            try:
                explanation = explain_case(experiment, row)
            except Exception as exc:
                st.error(f"SHAP could not calculate an explanation: {exc}")
                explanation = None
        if explanation is not None:
            st.markdown("#### Why did the model produce this output?")
            st.caption("Positive SHAP values move the model output toward class 1; negative values move it toward class 0. These are associations learned by this model, not causes.")
            shown = explanation.head(8).sort_values("SHAP value")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#bb4b45" if value > 0 else "#367a8b" for value in shown["SHAP value"]]
            ax.barh(shown["Feature"], shown["SHAP value"], color=colors)
            ax.axvline(0, color="#555", linewidth=.8)
            ax.set_xlabel("SHAP contribution to class 1 output")
            ax.set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig, width="stretch", alt="Local SHAP contributions for the selected model output")
            plt.close(fig)
            st.dataframe(explanation.head(8).style.format({"SHAP value": "{:+.4f}"}), width="stretch", hide_index=True)

with tab_method:
    st.subheader("Method")
    st.markdown(
        """
        1. Load the attributed UCI Heart Disease dataset 45 copy and use the 13 selected Cleveland predictors.
        2. Convert the original outcome `num` to the synopsis target: `0 → class 0`, `1–4 → class 1`.
        3. Split data with stratification and a fixed seed. Imputation, encoding, and scaling are fitted inside each model pipeline.
        4. Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Select on stratified training-partition cross-validation ROC-AUC.
        5. Report metrics on the held-out test partition and explain a selected input with model-agnostic SHAP.
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
        - This is a small historical dataset with 303 records, not a representative or prospectively collected clinical cohort.
        - The binary label is a project-specific grouping of the dataset's `num` field. It is not a diagnosis.
        - Test metrics can vary substantially with a small sample. The model probability is not presented as calibrated clinical risk.
        - Feature contribution explanations describe model behaviour; they do not prove causation, fairness, safety, or correctness.
        - No real patient records, clinical workflow, external validation, regulatory review, or clinician oversight are included.
        """
    )
    st.subheader("Dataset record")
    st.markdown("UCI Machine Learning Repository, Heart Disease (dataset ID 45), Cleveland subset; DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).")

st.divider()
st.caption("Project PCS 26 / 125 · Integral University, Lucknow · Zaid Hasan Khan, Raza Khan, Soyab Khan · Supervisor: Dr. S. H. Abbas Rizvi")
