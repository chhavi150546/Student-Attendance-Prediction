"""Run with:  python -m pytest -q   (from the project folder)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import attendance_predictor as ap  # noqa: E402

BRIEF = {"Lectures_Delivered": 100, "Lectures_Attended": 62, "Medical_Leave": 4, "Duty_Leave": 2}


@pytest.fixture(scope="module")
def system():
    df = ap.preprocess(ap.generate_synthetic_dataset())
    Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te, scaler = ap.split_and_scale(df)
    clf, reg = ap.train_models(Xtr, yc_tr, yr_tr)
    return clf, reg, scaler, (Xte, yc_te)


def test_inputs_are_only_the_four_lecture_counts():
    assert ap.INPUT_FIELDS == ["Lectures_Delivered", "Lectures_Attended", "Medical_Leave", "Duty_Leave"]
    assert not any("Internal" in f or "Marks" in f for f in ap.INPUT_FIELDS + ap.FEATURES)


def test_classifier_accuracy_reasonable(system):
    clf, _, _, (Xte, yc_te) = system
    assert clf.score(Xte, yc_te) > 0.8


def test_brief_example_is_high_risk(system):
    clf, reg, scaler, _ = system
    r = ap.predict_attendance(BRIEF, clf, reg, scaler)
    assert r["risk_level"] == "HIGH" and r["shortage"] is True
    assert r["current_attendance"] == 68.0 and r["must_attend"] == 28
    assert "At Risk" in r["status"] and 0 <= r["predicted_attendance"] <= 100


def test_strong_student_is_low_risk(system):
    clf, reg, scaler, _ = system
    good = {"Lectures_Delivered": 100, "Lectures_Attended": 95, "Medical_Leave": 2, "Duty_Leave": 1}
    r = ap.predict_attendance(good, clf, reg, scaler)
    assert r["risk_level"] == "LOW" and r["shortage"] is False and r["can_skip"] > 0


@pytest.mark.parametrize("bad", [
    {**BRIEF, "Lectures_Attended": 150},                 # more attended than delivered
    {**BRIEF, "Medical_Leave": 40, "Duty_Leave": 5},     # attended + leaves > delivered
    {**BRIEF, "Medical_Leave": -1},
    {**BRIEF, "Lectures_Delivered": 0},
    {**BRIEF, "Lectures_Attended": 61.5},                # not a whole number
    {**BRIEF, "Duty_Leave": "abc"},
])
def test_invalid_input_rejected(system, bad):
    clf, reg, scaler, _ = system
    with pytest.raises(ValueError):
        ap.predict_attendance(bad, clf, reg, scaler)


def test_missing_input_rejected(system):
    clf, reg, scaler, _ = system
    with pytest.raises(ValueError):
        ap.predict_attendance({"Medical_Leave": 3}, clf, reg, scaler)


def test_string_numbers_from_web_forms_work(system):
    clf, reg, scaler, _ = system
    r = ap.predict_attendance({k: str(v) for k, v in BRIEF.items()}, clf, reg, scaler)
    assert r["current_attendance"] == 68.0


@pytest.mark.parametrize("p,level", [(0.9, "HIGH"), (0.5, "MEDIUM"), (0.1, "LOW")])
def test_risk_buckets(p, level):
    assert ap.determine_risk_level(p) == level


def test_percentage_counts_leaves_as_present():
    s = ap.attendance_stats(80, 70, 2, 1)
    assert s["pct"] == 91.25 and s["raw_pct"] == 87.5 and s["absent"] == 7 and s["can_skip"] == 17


def test_exactly_at_requirement_is_not_a_shortage():
    s = ap.attendance_stats(100, 75, 0, 0)
    assert s["pct"] == 75.0 and s["shortage"] is False and s["can_skip"] == 0 and s["must_attend"] == 0


@pytest.mark.parametrize("d,a,m,u", [(100, 62, 4, 2), (37, 20, 1, 0), (200, 120, 10, 8), (9, 4, 0, 0), (60, 51, 3, 2)])
def test_must_attend_and_can_skip_are_exact(d, a, m, u):
    req = ap.REQUIRED_ATTENDANCE
    s = ap.attendance_stats(d, a, m, u)
    counted = s["counted"]
    if s["shortage"]:
        n = s["must_attend"]
        assert (counted + n) * 100 >= req * (d + n)                    # attending n in a row is enough...
        assert n == 0 or (counted + n - 1) * 100 < req * (d + n - 1)   # ...and n-1 would not be
    else:
        k = s["can_skip"]
        assert counted * 100 >= req * (d + k)                          # missing k more is still fine...
        assert counted * 100 < req * (d + k + 1)                       # ...missing one more is not
