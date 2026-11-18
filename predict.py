"""
Command-line Prediction System.

Give the four lecture counts as options:

    python predict.py --delivered 100 --attended 62 --medical 4 --duty 2

or run it with no options and it will ask for each value:

    python predict.py

If no trained model exists yet, one is trained first (takes a few seconds).
"""

import argparse
import sys

from attendance_predictor import (
    INPUT_FIELDS, FIELD_LIMITS,
    load_trained_system, run_pipeline,
    predict_attendance, display_prediction, validate_student_input,
    make_stdout_safe,
)

# (feature, command-line flag, prompt wording)
INPUTS = [
    ("Lectures_Delivered", "--delivered", "Lectures delivered so far"),
    ("Lectures_Attended",  "--attended",  "Lectures attended"),
    ("Medical_Leave",      "--medical",   "Lectures on medical leave"),
    ("Duty_Leave",         "--duty",      "Lectures on duty leave"),
]


def ask(feature: str, label: str) -> float:
    """Prompts until the person enters a number inside the valid range."""
    low, high = FIELD_LIMITS[feature]
    while True:
        raw = input(f"  {label} [{low:g}-{high:g}]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("    Please enter a number.")
            continue
        if value != int(value):
            print("    Please enter a whole number of lectures.")
            continue
        if not low <= value <= high:
            print(f"    Must be between {low:g} and {high:g}.")
            continue
        return value


def main() -> None:
    make_stdout_safe()

    parser = argparse.ArgumentParser(
        description="Predict a student's future attendance and attendance-risk level."
    )
    for feature, flag, label in INPUTS:
        parser.add_argument(flag, dest=feature, type=float, metavar="VALUE", help=label)
    args = parser.parse_args()

    student = {feature: getattr(args, feature) for feature in INPUT_FIELDS}

    try:
        if any(value is None for value in student.values()):
            print("Enter the student's details:\n")
            for feature, _, label in INPUTS:
                if student[feature] is None:
                    student[feature] = ask(feature, label)
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")

    try:
        validate_student_input(student)
    except ValueError as err:
        parser.error(str(err))

    try:
        clf, reg, scaler = load_trained_system()
    except FileNotFoundError:
        print("No trained model found yet - training one now (takes a few seconds)...\n")
        import matplotlib
        matplotlib.use("Agg")
        run_pipeline()
        print("\n" + "=" * 60 + "\n")
        clf, reg, scaler = load_trained_system()

    result = predict_attendance(student, clf, reg, scaler)
    print()
    display_prediction(result)


if __name__ == "__main__":
    main()
