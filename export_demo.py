"""
Refreshes the interactive HTML demo with the currently trained model.

The demo (attendance_prediction_demo.html) runs entirely in the browser, so
it carries a copy of the trained coefficients. After retraining -- for
example on your own data -- run:

    python export_demo.py

to rewrite that copy so the demo matches your model.
"""

import json
import re
import sys

from attendance_predictor import (
    BASE_DIR, FEATURES, REQUIRED_ATTENDANCE, load_trained_system,
)

HTML_PATH = BASE_DIR / "attendance_prediction_demo.html"
START, END = "/* MODEL_START */", "/* MODEL_END */"


def main() -> None:
    try:
        clf, reg, scaler = load_trained_system()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}")

    model = {
        "features": FEATURES,
        "scalerMean": scaler.mean_.tolist(),
        "scalerScale": scaler.scale_.tolist(),
        "logregCoef": clf.coef_[0].tolist(),
        "logregIntercept": float(clf.intercept_[0]),
        "linregCoef": reg.coef_.tolist(),
        "linregIntercept": float(reg.intercept_),
        "required": REQUIRED_ATTENDANCE,
    }
    block = f"{START}\n  const MODEL = {json.dumps(model, indent=2)};\n  {END}"

    html = HTML_PATH.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(html):
        sys.exit(f"Error: could not find the {START} ... {END} markers in {HTML_PATH.name}")

    HTML_PATH.write_text(pattern.sub(lambda _: block, html, count=1), encoding="utf-8")
    print(f"Updated {HTML_PATH.name} with the current trained model.")


if __name__ == "__main__":
    main()
