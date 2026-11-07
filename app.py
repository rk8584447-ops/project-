"""Streamlit interface for the explainable heart-disease academic prototype."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import RocCurveDisplay

from src.data import feature_metadata, load_heart_data
from src.modeling import explain_case, global_permutation_importance, train_experiment

st.set_page_config(page_title="Explainable CDSS | Heart Disease", page_icon="🫀", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {max-width: 1250px; padding-top: 2rem;}
    .project-kicker {color:#557086; font-size:.82rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;}
    .risk-note {padding:1rem 1.1rem; background:#fff6e8; border-left:5px solid #d98c10; border-radius:.4rem; color:#573b12;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="project-kicker">Academic prototype · UCI Heart Disease dataset</div>', unsafe_allow_html=True)
st.title("Explainable Clinical Decision Support")
st.write("Compare classical classifiers and inspect the feature contributions behind one model output.")
st.markdown(
    '<div class="risk-note"><strong>For academic demonstration only.</strong> This prototype is not clinically validated and must not be used for diagnosis, treatment, screening, or patient care. Do not enter real patient information. A model explanation does not establish that a prediction is correct.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading public data and training model comparison…")
def get_experiment():
    X, y = load_heart_data()
    return X, y, train_experiment(X, y)


try:
    X, y, experiment = get_experiment()
except Exception as exc:
    st.error(f"The project could not prepare the dataset: {exc}")
    st.info("Check internet access for the first UCI download and install dependencies from requirements.txt.")
    st.stop()

tab_overview, tab_predict, tab_method = st.tabs(["Model overview", "Explore one case", "Method & limitations"])

with tab_overview:
    st.subheader("Held-out model comparison")
    st.caption(
        f"{len(X)} usable records · {X.shape[1]} predictors · stratified {len(experiment['y_test'])}-record test set · "
        f"{experiment['cv_folds']}-fold CV on training data. Selected model: {experiment['best_name']} by mean CV ROC-AUC."
    )
    st.dataframe(
        experiment["results"].style.format({column: "{:.3f}" for column in experiment["results"].columns if column != "Model"}),
        width="stretch",
        hide_index=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Selected model: confusion matrix**")
        cm = experiment["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), width="stretch")
        st.caption("Target 0/1 reflects the UCI dataset grouping (num=0 vs num=1–4), not a clinical determination.")
    with right:
        st.markdown("**Selected model: ROC curve**")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        RocCurveDisplay.from_predictions(experiment["y_test"], experiment["probabilities"], ax=ax, name=experiment["best_name"])
        ax.plot([0, 1], [0, 1], linestyle="--", color="#8b949e", linewidth=1)
        ax.set_title("Held-out test data")
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="ROC curve for the selected model on the held-out test split")
        plt.close(fig)
    st.markdown("**Global feature importance**")
    st.caption("Permutation importance measures the change in held-out ROC-AUC when one input feature is shuffled. It is a model diagnostic, not a causal effect.")
    if st.button("Calculate global importance", key="global_importance"):
        with st.spinner("Calculating permutation importance…"):
            importance = global_permutation_importance(experiment)
        st.dataframe(importance.style.format({"Importance (mean ROC-AUC decrease)": "{:.4f}", "Std": "{:.4f}"}), width="stretch", hide_index=True)

with tab_predict:
    st.subheader("Inspect a structured input")
    st.caption("Inputs are interpreted using the dataset's numeric codes. This form is for exploring the model only; it does not provide medical advice.")
    metadata = feature_metadata()
    with st.form("case_form"):
        entered = {}
        columns = st.columns(3)
        for index, feature in enumerate(X.columns):
            meta = metadata.get(feature, {"label": feature, "min": float(X[feature].min()), "max": float(X[feature].max()), "default": float(X[feature].median()), "step": 1.0})
            with columns[index % 3]:
                if "options" in meta:
                    options = meta["options"]
                    entered[feature] = st.selectbox(meta["label"], options, index=options.index(meta["default"]))
                else:
                    entered[feature] = st.number_input(
                        f"{meta['label']} ({meta.get('unit', '')})".strip(),
                        min_value=float(meta["min"]), max_value=float(meta["max"]),
                        value=float(meta["default"]), step=float(meta["step"]),
                    )
        submitted = st.form_submit_button("Generate prediction and explanation", type="primary")

    if submitted:
        row = pd.DataFrame([{feature: entered[feature] for feature in X.columns}], columns=X.columns)
        probability = float(experiment["best_model"].predict_proba(row)[0, 1])
        predicted_class = int(experiment["best_model"].predict(row)[0])
        first, second, third = st.columns(3)
        first.metric("Model output", "Dataset class 1" if predicted_class else "Dataset class 0")
        second.metric("Model probability for class 1", f"{probability:.1%}")
        third.metric("Selected model", experiment["best_name"])
        st.caption("Probability is the classifier's output on this academic dataset; it is not a calibrated personal risk estimate.")
        with st.spinner("Calculating local SHAP contributions… this may take a short while."):
            try:
                explanation = explain_case(experiment, row)
            except Exception as exc:
                st.error(f"SHAP could not calculate an explanation: {exc}")
                explanation = None
        if explanation is not None:
            st.markdown("**Largest local feature contributions**")
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
