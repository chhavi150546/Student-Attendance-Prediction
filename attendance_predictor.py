"""
Student Attendance Prediction System
=====================================
Predicts a student's likely future attendance percentage and classifies
their risk of falling below the required attendance threshold.

Pipeline (matches the project's architecture diagram):

    Student Data -> Data Preprocessing -> Feature Selection -> Train/Test Split
    -> ML Model (Logistic Regression + Linear Regression) -> Model Evaluation
    -> Trained ML Model -> Prediction System -> Student Input -> Prediction
       (Attendance Status / Expected % / Risk Level)

What a student records (the only inputs):
    Lectures_Delivered   total lectures held so far
    Lectures_Attended    lectures the student was present for
    Medical_Leave        lectures missed on approved medical leave
    Duty_Leave           lectures missed on approved duty leave (sports, NSS, events...)
  From these the system works out the attendance percentage, the number of
  lectures the student can still miss, and the model's risk prediction.

Run it:
    python attendance_predictor.py                 train, evaluate, save, demo
    python attendance_predictor.py --data my.csv   train on your own records

Why two models?
  The two outputs the project needs are different in kind:
    - Predicted Attendance (%)   a continuous number     -> Linear Regression
    - Risk Level (LOW/MED/HIGH)  a probability-backed class -> Logistic Regression
  Logistic Regression is the baseline classifier the project asks for. Its
  predict_proba() output is what the LOW / MEDIUM / HIGH risk levels are
  bucketed from -- the point isn't only the class, it's the confidence
  behind it.

About the data:
  No real student records were supplied, so by default this module generates
  a realistic *synthetic* dataset to build and evaluate the pipeline. To use
  real records, pass --data (see load_dataset() for the required columns).
"""

import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, r2_score,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# The minimum attendance % your institution requires. This is what "sufficient
# attendance" means to the classifier: it labels the synthetic data, and the
# model learns that boundary. If your rule differs, change it and retrain.
REQUIRED_ATTENDANCE = 75.0

# Do approved leaves count as "present" when the percentage is worked out?
#   percentage = (attended [+ medical] [+ duty]) / delivered * 100
# Most institutions count both; set either flag to False if yours does not.
COUNT_MEDICAL_LEAVE = True
COUNT_DUTY_LEAVE = True

# What a student types in (whole numbers of lectures).
INPUT_FIELDS = [
    "Lectures_Delivered",
    "Lectures_Attended",
    "Medical_Leave",
    "Duty_Leave",
]

# What the models actually learn from -- derived from the inputs above, so the
# model sees *rates* (comparable across students) rather than raw counts.
FEATURES = [
    "Attendance_Pct",
    "Medical_Pct",
    "Duty_Pct",
    "Lectures_Delivered",
]

# Hard limits for a valid input value: (min, max)
FIELD_LIMITS = {
    "Lectures_Delivered": (1, 2000),
    "Lectures_Attended": (0, 2000),
    "Medical_Leave": (0, 2000),
    "Duty_Leave": (0, 2000),
}

# Paths are anchored to this file, so every command works from any directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_DATA_PATH = DATA_DIR / "student_attendance_data.csv"
MODEL_FILES = ("logistic_model.joblib", "linear_model.joblib", "scaler.joblib")


def make_stdout_safe():
    """Windows consoles/pipes can use a legacy code page that can't print the
    status emoji; replace unprintable characters instead of crashing."""
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Attendance arithmetic (exact, no machine learning involved)
# --------------------------------------------------------------------------
def attendance_stats(delivered, attended, medical=0, duty=0, required=None) -> dict:
    """
    Works out everything that follows directly from the four lecture counts.

        counted   lectures that count as present (attended, plus approved leaves)
        pct       counted / delivered * 100
        raw_pct   attended / delivered * 100  (the student's own presence, no leaves)
        absent    lectures missed without an approved leave
        can_skip  how many MORE lectures can be missed while staying at/above the
                  requirement (0 when already below it)
        must_attend  how many lectures in a row must be attended, with none missed,
                  to get back to the requirement (0 when already at/above it)
    """
    required = REQUIRED_ATTENDANCE if required is None else required
    delivered, attended, medical, duty = int(delivered), int(attended), int(medical), int(duty)
    counted = attended + (medical if COUNT_MEDICAL_LEAVE else 0) + (duty if COUNT_DUTY_LEAVE else 0)
    shortage = counted * 100 < required * delivered          # exact integer-style comparison

    can_skip = 0 if shortage else max(0, int(math.floor(counted * 100 / required - delivered + 1e-9)))
    if not shortage:
        must_attend = 0
    elif required >= 100:
        must_attend = None                                   # can never be reached by attending alone
    else:
        must_attend = max(0, int(math.ceil((required * delivered - 100 * counted) / (100 - required) - 1e-9)))

    return {
        "delivered": delivered, "attended": attended, "medical": medical, "duty": duty,
        "absent": delivered - attended - medical - duty,
        "counted": counted,
        "pct": round(counted / delivered * 100, 2),
        "raw_pct": round(attended / delivered * 100, 2),
        "shortage": shortage,
        "can_skip": can_skip,
        "must_attend": must_attend,
    }


def derive_features(delivered, attended, medical, duty) -> dict:
    """The four model features for one student, from the four lecture counts."""
    s = attendance_stats(delivered, attended, medical, duty)
    return {
        "Attendance_Pct": s["pct"],
        "Medical_Pct": 100 * s["medical"] / s["delivered"],
        "Duty_Pct": 100 * s["duty"] / s["delivered"],
        "Lectures_Delivered": float(s["delivered"]),
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised version of derive_features() for a whole table."""
    df = df.copy()
    counted = df["Lectures_Attended"] \
        + (df["Medical_Leave"] if COUNT_MEDICAL_LEAVE else 0) \
        + (df["Duty_Leave"] if COUNT_DUTY_LEAVE else 0)
    df["Attendance_Pct"] = (counted / df["Lectures_Delivered"] * 100).round(2)
    df["Medical_Pct"] = df["Medical_Leave"] / df["Lectures_Delivered"] * 100
    df["Duty_Pct"] = df["Duty_Leave"] / df["Lectures_Delivered"] * 100
    return df


# --------------------------------------------------------------------------
# 1. STUDENT DATA
# --------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples: int = 1200, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset of student attendance records.

    Each student has a latent "engagement" trait that pulls how often they
    attend up or down. Medical and duty leaves are independent of engagement
    (illness and college events happen to everyone). Noise is added to the
    future-attendance target so nothing is perfectly predictable.
    """
    rng = np.random.default_rng(random_state)
    engagement = rng.normal(0, 1, n_samples)

    delivered = rng.integers(20, 121, n_samples)
    own_rate = np.clip(0.64 + 0.13 * engagement + rng.normal(0, 0.04, n_samples), 0.15, 0.99)
    medical_rate = np.clip(rng.beta(1.2, 20, n_samples), 0, 0.30)
    duty_rate = np.clip(rng.beta(1.2, 25, n_samples), 0, 0.25)

    attended = np.rint(delivered * own_rate).astype(int)
    medical = np.rint(delivered * medical_rate).astype(int)
    duty = np.rint(delivered * duty_rate).astype(int)
    attended = attended - np.maximum(attended + medical + duty - delivered, 0)   # never exceed delivered

    counted_pct = (attended + medical + duty) / delivered * 100
    own_pct = attended / delivered * 100
    noise = rng.normal(0, 5, n_samples)
    future_attendance = np.clip(0.60 * counted_pct + 0.30 * own_pct + 3.0 * engagement + 10 + noise, 0, 100)
    sufficient = (future_attendance >= REQUIRED_ATTENDANCE).astype(int)

    return pd.DataFrame({
        "Lectures_Delivered": delivered,
        "Lectures_Attended": attended,
        "Medical_Leave": medical,
        "Duty_Leave": duty,
        "Future_Attendance": future_attendance.round(1),
        "Sufficient_Attendance": sufficient,
    })


def load_dataset(path) -> pd.DataFrame:
    """
    Loads real student records from a CSV file.

    Required columns:
        Lectures_Delivered, Lectures_Attended, Medical_Leave, Duty_Leave
                              -- the four inputs (whole numbers of lectures)
        Future_Attendance     -- the student's attendance % in the NEXT period
    Optional column:
        Sufficient_Attendance -- 1 if that future attendance was enough,
                                 else 0. If omitted it is derived from
                                 Future_Attendance >= REQUIRED_ATTENDANCE.
    """
    df = pd.read_csv(path)
    needed = INPUT_FIELDS + ["Future_Attendance"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    if "Sufficient_Attendance" not in df.columns:
        df["Sufficient_Attendance"] = (df["Future_Attendance"] >= REQUIRED_ATTENDANCE).astype(int)
    return df


# --------------------------------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with no target, fills missing inputs with the column median,
    clips every input to its valid range, drops rows whose counts don't add up
    (attended + medical + duty > delivered), then derives the model features."""
    df = df.copy()
    df = df.dropna(subset=["Future_Attendance", "Sufficient_Attendance"])
    df[INPUT_FIELDS] = df[INPUT_FIELDS].apply(lambda col: col.fillna(col.median()))
    for field, (low, high) in FIELD_LIMITS.items():
        df[field] = df[field].clip(low, high)
    df[INPUT_FIELDS] = df[INPUT_FIELDS].round().astype(int)
    consistent = (df["Lectures_Attended"] + df["Medical_Leave"] + df["Duty_Leave"]) <= df["Lectures_Delivered"]
    df = df[consistent]
    return add_features(df)


# --------------------------------------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------------------------------------
def feature_selection_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each feature's statistical relationship with both targets (F-test).
    Shows how strongly each model feature relates to the outcome.
    """
    X = df[FEATURES]
    f_cls, p_cls = f_classif(X, df["Sufficient_Attendance"])
    f_reg, p_reg = f_regression(X, df["Future_Attendance"])
    return pd.DataFrame({
        "Feature": FEATURES,
        "F-score (class.)": f_cls.round(1),
        "p-value (class.)": p_cls.round(4),
        "F-score (regr.)": f_reg.round(1),
        "p-value (regr.)": p_reg.round(4),
    })


# --------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT (+ scaling)
# --------------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    X = df[FEATURES].values
    y_class = df["Sufficient_Attendance"].values
    y_reg = df["Future_Attendance"].values

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg, test_size=test_size, random_state=RANDOM_STATE, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, yc_train, yc_test, yr_train, yr_test, scaler


# --------------------------------------------------------------------------
# 5. MACHINE LEARNING MODEL
# --------------------------------------------------------------------------
def train_models(X_train, yc_train, yr_train):
    clf = LogisticRegression(random_state=RANDOM_STATE)
    clf.fit(X_train, yc_train)

    reg = LinearRegression()
    reg.fit(X_train, yr_train)

    return clf, reg


# --------------------------------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------------------------------
def evaluate_classifier(clf, X_test, yc_test, save_path=None):
    """Accuracy, precision, recall, F1 and a confusion-matrix figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(yc_test, y_pred),
        "precision": precision_score(yc_test, y_pred),
        "recall": recall_score(yc_test, y_pred),
        "f1": f1_score(yc_test, y_pred),
    }
    cm = confusion_matrix(yc_test, y_pred)
    report_text = classification_report(yc_test, y_pred, target_names=["Insufficient", "Sufficient"])

    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Insufficient", "Sufficient"],
        yticklabels=["Insufficient", "Sufficient"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Attendance Sufficiency")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)

    return metrics, cm, report_text, fig


def evaluate_regressor(reg, X_test, yr_test):
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(yr_test, y_pred)
    rmse = float(np.sqrt(np.mean((yr_test - y_pred) ** 2)))
    r2 = r2_score(yr_test, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# --------------------------------------------------------------------------
# 7. TRAINED ML MODEL (save / load)
# --------------------------------------------------------------------------
def save_trained_system(clf, reg, scaler):
    """Persists the fitted classifier, regressor and scaler to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [MODELS_DIR / name for name in MODEL_FILES]
    for obj, path in zip((clf, reg, scaler), paths):
        joblib.dump(obj, path)
    return paths


def load_trained_system():
    """Loads (classifier, regressor, scaler) saved by save_trained_system()."""
    paths = [MODELS_DIR / name for name in MODEL_FILES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Trained model not found (missing: {', '.join(missing)}). "
            "Train it first with:  python attendance_predictor.py"
        )
    clf, reg, scaler = (joblib.load(p) for p in paths)
    return clf, reg, scaler


# --------------------------------------------------------------------------
# 8. PREDICTION SYSTEM
# --------------------------------------------------------------------------
def _whole_number(name: str, raw) -> int:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name.replace('_', ' ')} must be a number (got {raw!r})")
    if value != int(value):
        raise ValueError(f"{name.replace('_', ' ')} must be a whole number of lectures (got {value:g})")
    return int(value)


def validate_student_input(student_input: dict) -> None:
    """Raises ValueError with a clear message if an input is missing or invalid."""
    missing = [f for f in INPUT_FIELDS if f not in student_input]
    if missing:
        raise ValueError("Missing input(s): " + ", ".join(missing))
    values = {f: _whole_number(f, student_input[f]) for f in INPUT_FIELDS}
    for field, (low, high) in FIELD_LIMITS.items():
        if not low <= values[field] <= high:
            raise ValueError(f"{field.replace('_', ' ')} must be between {low:g} and {high:g} (got {values[field]:g})")
    used = values["Lectures_Attended"] + values["Medical_Leave"] + values["Duty_Leave"]
    if used > values["Lectures_Delivered"]:
        raise ValueError(
            f"Attended ({values['Lectures_Attended']}) + medical leave ({values['Medical_Leave']}) + "
            f"duty leave ({values['Duty_Leave']}) = {used}, which is more than the "
            f"{values['Lectures_Delivered']} lectures delivered."
        )


def determine_risk_level(prob_insufficient: float) -> str:
    """Buckets the logistic regression's probability into three risk tiers."""
    if prob_insufficient >= 0.66:
        return "HIGH"
    if prob_insufficient >= 0.33:
        return "MEDIUM"
    return "LOW"


def _lectures(n: int) -> str:
    return f"{n} lecture" + ("" if n == 1 else "s")


def generate_recommendation(risk_level: str, stats: dict) -> str:
    """Rule-based advice (not a model output): the exact lecture numbers from
    attendance_stats(), framed by the model's risk level."""
    req = f"{REQUIRED_ATTENDANCE:g}%"
    if stats["shortage"]:
        need = stats["must_attend"]
        text = (f"You are below {req}. Attend your next {_lectures(need)} in a row, "
                f"without missing any, to get back to {req}." if need is not None
                else f"You are below {req}; speak to your teacher about how to recover.")
    elif risk_level == "HIGH":
        text = (f"You're above {req} right now but the trend is risky - you can only afford to miss "
                f"{_lectures(stats['can_skip'])} more. Avoid absences.")
    elif risk_level == "MEDIUM":
        text = (f"Your attendance is borderline - you can miss at most {_lectures(stats['can_skip'])} "
                f"more before dropping below {req}.")
    else:
        text = (f"You're on track. You can still miss {_lectures(stats['can_skip'])} "
                f"without falling below {req}.")

    leaves = stats["medical"] + stats["duty"]
    if leaves and stats["raw_pct"] < REQUIRED_ATTENDANCE <= stats["pct"]:
        text += (f" Note: approved leaves are covering {_lectures(leaves)} - your own presence is "
                 f"{stats['raw_pct']:g}%, so don't rely on leaves.")
    return text


def predict_attendance(student_input: dict, clf, reg, scaler) -> dict:
    """
    The Prediction System.

    Takes one student's four lecture counts (Lectures_Delivered, Lectures_Attended,
    Medical_Leave, Duty_Leave), works out the attendance percentage, scales the
    derived features with the fitted scaler, and returns the predicted future
    attendance %, risk level (LOW / MEDIUM / HIGH), status and a recommendation.
    Raises ValueError for missing, non-whole or inconsistent inputs.
    """
    validate_student_input(student_input)
    counts = {f: _whole_number(f, student_input[f]) for f in INPUT_FIELDS}
    stats = attendance_stats(counts["Lectures_Delivered"], counts["Lectures_Attended"],
                             counts["Medical_Leave"], counts["Duty_Leave"])
    features = derive_features(counts["Lectures_Delivered"], counts["Lectures_Attended"],
                               counts["Medical_Leave"], counts["Duty_Leave"])
    x_scaled = scaler.transform(np.array([[features[f] for f in FEATURES]]))

    predicted_attendance = float(np.clip(reg.predict(x_scaled)[0], 0, 100))
    prob_insufficient = 1 - clf.predict_proba(x_scaled)[0][1]

    risk_level = determine_risk_level(prob_insufficient)
    safe = risk_level == "LOW"

    return {
        "current_attendance": stats["pct"],
        "own_attendance": stats["raw_pct"],
        "delivered": stats["delivered"], "attended": stats["attended"],
        "medical": stats["medical"], "duty": stats["duty"], "absent": stats["absent"],
        "shortage": stats["shortage"],
        "can_skip": stats["can_skip"], "must_attend": stats["must_attend"],
        "predicted_attendance": round(predicted_attendance, 1),
        "required_attendance": REQUIRED_ATTENDANCE,
        "risk_level": risk_level,
        "risk_probability_pct": round(prob_insufficient * 100, 1),
        "status": ("\u2705 Safe" if safe else "\u26A0\uFE0F At Risk"),
        "recommendation": generate_recommendation(risk_level, stats),
    }


def display_prediction(result: dict) -> None:
    rows = [
        ("Lectures delivered:", f"{result['delivered']}"),
        ("Attended:", f"{result['attended']}"),
        ("Medical / Duty leave:", f"{result['medical']} / {result['duty']}"),
        ("Current Attendance:", f"{result['current_attendance']:g}%"),
        ("Predicted Attendance:", f"{result['predicted_attendance']:g}%"),
        ("Required Attendance:", f"{result['required_attendance']:g}%"),
    ]
    for label, value in rows:
        print(f"{label:<23}{value}")
    print()
    print(f"Risk Level: {result['risk_level']}  "
          f"({result['risk_probability_pct']:g}% chance of falling short)")
    print(f"Status: {result['status']}")
    print()
    print("Recommendation:")
    print(textwrap.fill(result["recommendation"], width=72))


# --------------------------------------------------------------------------
# Full pipeline runner
# --------------------------------------------------------------------------
def run_pipeline(data_path=None):
    """Runs every stage end to end and saves the data, models and plots."""
    import matplotlib.pyplot as plt

    print("STEP 1 - Student data")
    if data_path:
        df = load_dataset(data_path)
        print(f"  Loaded {len(df)} student records from {data_path}")
    else:
        df = generate_synthetic_dataset()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DEFAULT_DATA_PATH, index=False)
        print(f"  Generated {len(df)} synthetic student records -> {DEFAULT_DATA_PATH.relative_to(BASE_DIR)}")

    print("\nSTEP 2 - Preprocessing")
    df = preprocess(df)
    print(f"  {len(df)} usable rows, {int(df[FEATURES].isnull().sum().sum())} missing feature values left")
    print(f"  Share of students labelled 'sufficient': {df['Sufficient_Attendance'].mean():.1%}")
    if df["Sufficient_Attendance"].nunique() < 2:
        raise ValueError(
            "Every student has the same Sufficient_Attendance value, so a classifier "
            "can't be trained. Check REQUIRED_ATTENDANCE and the Future_Attendance values."
        )

    print("\nSTEP 3 - Feature selection (F-test against both targets)")
    print(feature_selection_report(df).to_string(index=False))

    print("\nSTEP 4 - Train/test split (80/20, stratified) + scaling")
    X_train, X_test, yc_train, yc_test, yr_train, yr_test, scaler = split_and_scale(df)
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("\nSTEP 5 - Training Logistic Regression (risk) + Linear Regression (expected %)")
    clf, reg = train_models(X_train, yc_train, yr_train)

    print("\nSTEP 6 - Evaluation on the held-out test set")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    metrics, cm, report_text, fig = evaluate_classifier(clf, X_test, yc_test, save_path=cm_path)
    plt.close(fig)
    print("  Logistic Regression (classifier):")
    for name, value in metrics.items():
        print(f"    {name.capitalize():<10}: {value:.3f}")
    print("  Confusion matrix [[TN FP] [FN TP]]:", cm.tolist())
    print(f"  (plot saved to {cm_path.relative_to(BASE_DIR)})")
    print("\n  Classification report:")
    print(textwrap.indent(report_text, "  "))
    reg_metrics = evaluate_regressor(reg, X_test, yr_test)
    print("  Linear Regression (expected attendance %):")
    for name, value in reg_metrics.items():
        print(f"    {name.upper():<5}: {value:.3f}")

    print("\nSTEP 7 - Saving the trained model")
    for path in save_trained_system(clf, reg, scaler):
        print(f"  Saved {path.relative_to(BASE_DIR)}")

    print("\nSTEP 8 - Prediction system demo (student input -> prediction)")
    example_student = {
        "Lectures_Delivered": 100,
        "Lectures_Attended": 62,
        "Medical_Leave": 4,
        "Duty_Leave": 2,
    }
    result = predict_attendance(example_student, clf, reg, scaler)
    print()
    display_prediction(result)

    return {
        "df": df, "clf": clf, "reg": reg, "scaler": scaler,
        "metrics": metrics, "cm": cm, "reg_metrics": reg_metrics,
        "demo_result": result,
    }


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")   # script mode: write plots to files, never open windows
    make_stdout_safe()

    parser = argparse.ArgumentParser(
        description="Train the attendance prediction models, evaluate them, and run a demo prediction."
    )
    parser.add_argument(
        "--data", metavar="CSV",
        help="train on your own student records instead of the synthetic dataset",
    )
    args = parser.parse_args()
    try:
        run_pipeline(data_path=args.data)
    except (ValueError, FileNotFoundError) as err:
        sys.exit(f"Error: {err}")
