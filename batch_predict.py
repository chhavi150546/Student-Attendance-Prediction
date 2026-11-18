"""
Predict for a whole class at once.

    python batch_predict.py students.csv                 -> students_predictions.csv
    python batch_predict.py students.csv -o result.csv

The input CSV needs the four columns:
    Lectures_Delivered, Lectures_Attended, Medical_Leave, Duty_Leave
(any extra columns, such as Name or Roll_No, are kept in the output).
Rows with missing/invalid values are not dropped: they get an 'Error' message.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from attendance_predictor import (
    INPUT_FIELDS, load_trained_system, predict_attendance, make_stdout_safe,
)


def main() -> None:
    make_stdout_safe()
    parser = argparse.ArgumentParser(description="Predict attendance for many students from a CSV.")
    parser.add_argument("csv", help="input CSV with the four lecture-count columns")
    parser.add_argument("-o", "--output", help="output CSV (default: <input>_predictions.csv)")
    args = parser.parse_args()

    src = Path(args.csv)
    if not src.exists():
        sys.exit(f"Error: file not found: {src}")
    df = pd.read_csv(src)
    missing = [c for c in INPUT_FIELDS if c not in df.columns]
    if missing:
        sys.exit("Error: missing column(s): " + ", ".join(missing))

    try:
        clf, reg, scaler = load_trained_system()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}")

    cols = ["Attendance_Pct", "Predicted_Attendance", "Risk_Level", "Risk_Probability_Pct", "Status", "Recommendation", "Error"]
    out = {c: [] for c in cols}
    for _, row in df.iterrows():
        try:
            r = predict_attendance({f: row[f] for f in INPUT_FIELDS}, clf, reg, scaler)
            vals = [r["current_attendance"], r["predicted_attendance"], r["risk_level"], r["risk_probability_pct"],
                    "At Risk" if r["risk_level"] != "LOW" else "Safe", r["recommendation"], ""]
        except ValueError as err:
            vals = ["", "", "", "", "", "", str(err)]
        for c, v in zip(cols, vals):
            out[c].append(v)

    result = pd.concat([df, pd.DataFrame(out)], axis=1)
    dest = Path(args.output) if args.output else src.with_name(src.stem + "_predictions.csv")
    result.to_csv(dest, index=False)

    ok = result[result["Error"] == ""]
    print(f"Wrote {len(result)} rows to {dest}")
    print(f"  High risk: {(ok['Risk_Level'] == 'HIGH').sum()} | Medium: {(ok['Risk_Level'] == 'MEDIUM').sum()} "
          f"| Low: {(ok['Risk_Level'] == 'LOW').sum()} | Errors: {len(result) - len(ok)}")


if __name__ == "__main__":
    main()
