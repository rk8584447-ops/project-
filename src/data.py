"""Dataset access and target preparation for the UCI Heart Disease data."""

from pathlib import Path
import hashlib

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_FILE = DATA_DIR / "heart_disease.csv"
TARGET_COLUMN = "num"


def dataset_signature(X: pd.DataFrame, y: pd.Series) -> str:
    """Return a stable digest for the prepared features and target."""
    prepared = X.copy()
    prepared["__project_target__"] = y.to_numpy()
    hashed = pd.util.hash_pandas_object(prepared, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def load_heart_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI dataset 45, caching the source table locally after first fetch."""
    if CACHE_FILE.exists() and not refresh:
        frame = pd.read_csv(CACHE_FILE)
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"Cached file must contain a '{TARGET_COLUMN}' target column.")
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to download the UCI dataset.") from exc

        try:
            dataset = fetch_ucirepo(id=45)
            frame = dataset.data.features.copy()
            target = dataset.data.targets.copy()
            # The UCI API names this outcome 'num'. Keep an explicit fallback for
            # metadata revisions while refusing an ambiguous multi-target result.
            target_name = TARGET_COLUMN if TARGET_COLUMN in target.columns else target.columns[0]
            frame[TARGET_COLUMN] = target[target_name].values
        except Exception as exc:
            raise RuntimeError(
                "Could not fetch UCI dataset 45. Check internet access, then retry."
            ) from exc

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(CACHE_FILE, index=False)

    frame = frame.replace("?", pd.NA)
    y_raw = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = y_raw.notna()
    X = frame.loc[valid].drop(columns=[TARGET_COLUMN]).copy()
    # Coerce all UCI Cleveland fields to numeric; missing/invalid predictor cells
    # stay NaN and are imputed within the cross-validation/training pipeline.
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    y = (y_raw.loc[valid] > 0).astype("int64")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).rename("disease_finding")
    if len(X) < 30 or y.nunique() != 2:
        raise ValueError("The downloaded dataset does not contain enough valid binary-target rows.")
    return X, y


def feature_metadata() -> dict[str, dict[str, object]]:
    """Human-readable input labels, units, and sensible demo input ranges."""
    return {
        "age": {"label": "Age", "min": 18, "max": 100, "default": 55, "step": 1, "unit": "years", "help": "Age in years in the original Cleveland record."},
        "sex": {"label": "Sex (dataset field)", "options": [0, 1], "default": 1, "option_labels": {0: "Female", 1: "Male"}, "help": "Dataset codes: 0 = female; 1 = male."},
        "cp": {"label": "Chest pain type", "options": [1, 2, 3, 4], "default": 2, "option_labels": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}, "help": "Choose the category recorded in the dataset."},
        "trestbps": {"label": "Resting blood pressure", "min": 70, "max": 250, "default": 130, "step": 1, "unit": "mm Hg", "help": "Resting blood pressure on hospital admission, in mm Hg."},
        "chol": {"label": "Serum cholesterol", "min": 80, "max": 700, "default": 240, "step": 1, "unit": "mg/dL", "help": "Serum cholesterol recorded in mg/dL."},
        "fbs": {"label": "Fasting blood sugar", "options": [0, 1], "default": 0, "option_labels": {0: "≤ 120 mg/dL", 1: "> 120 mg/dL"}, "help": "Whether fasting blood sugar was greater than 120 mg/dL."},
        "restecg": {"label": "Resting ECG result", "options": [0, 1, 2], "default": 0, "option_labels": {0: "Normal", 1: "ST–T wave abnormality", 2: "Possible left ventricular hypertrophy"}, "help": "Select the recorded resting ECG category."},
        "thalach": {"label": "Maximum heart rate achieved", "min": 50, "max": 250, "default": 150, "step": 1, "unit": "bpm", "help": "Maximum heart rate achieved during the test, in beats per minute."},
        "exang": {"label": "Exercise-induced angina", "options": [0, 1], "default": 0, "option_labels": {0: "No", 1: "Yes"}, "help": "Whether exercise-induced angina was recorded."},
        "oldpeak": {"label": "ST depression (oldpeak)", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "unit": "dataset units", "help": "ST depression induced by exercise relative to rest."},
        "slope": {"label": "Peak exercise ST segment slope", "options": [1, 2, 3], "default": 2, "option_labels": {1: "Upsloping", 2: "Flat", 3: "Downsloping"}, "help": "Slope category recorded for the peak exercise ST segment."},
        "ca": {"label": "Major vessels (ca)", "options": [0, 1, 2, 3], "default": 0, "option_labels": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"}, "help": "Number of major vessels colored by fluoroscopy (0–3)."},
        "thal": {"label": "Thal result", "options": [3, 6, 7], "default": 3, "option_labels": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}, "help": "Select the thal category using the UCI dataset's recorded codes."},
    }
