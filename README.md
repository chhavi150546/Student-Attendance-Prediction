# Student Attendance Prediction System

A web app where **students** record their attendance and **teachers** see the whole
class. It works out each student's attendance percentage, how many lectures they
can still miss, and predicts whether they will clear the minimum required
attendance (75% by default).

A student enters just four whole numbers:

| Input                | Example |
|----------------------|---------|
| Lectures delivered   | 100     |
| Lectures attended    | 62      |
| Medical leave        | 4       |
| Duty leave           | 2       |

(Internal marks, study hours and assignments are **not** used - they don't affect
attendance.)

**Attendance %** = (attended + medical leave + duty leave) / delivered x 100
-> here `(62 + 4 + 2) / 100 = 68%`. If your institution does not count one of the
leave types as present, switch it off with `COUNT_MEDICAL_LEAVE` /
`COUNT_DUTY_LEAVE` at the top of `attendance_predictor.py`.

**Output:** attendance %, lectures the student can still miss (or must attend in a
row to recover), predicted attendance %, risk level (LOW / MEDIUM / HIGH),
status and a recommendation.

```
Current Attendance:    68%
Predicted Attendance:  68.6%
Required Attendance:   75%

Risk Level: HIGH  (87.5% chance of falling short)
Status: ⚠️ At Risk

Recommendation:
You are below 75%. Attend your next 28 lectures in a row, without missing
any, to get back to 75%.
```

> **About the data:** no real student records were available, so the project
> ships with a *synthetic* dataset (1,200 students) built to mirror typical
> relationships between attendance and leave patterns. It shows the pipeline works; it does
> not tell you how accurate the model will be on your own students. See
> [Use your own data](#use-your-own-data).

---

## What's inside

```
student-attendance-predictor/
├── README.md
├── requirements.txt
├── attendance_predictor.py              the whole pipeline + prediction system
├── predict.py                           command-line prediction for one student
├── app.py                               web app: students, teachers, class overview
├── templates/  static/                  HTML pages, CSS, JavaScript
├── instance/                            created on first run (database, secret key, teacher code)
├── batch_predict.py                     predict for a whole class from a CSV
├── tests/                               automated tests (model + web app)
├── export_demo.py                       refresh the HTML demo after retraining
├── Student_Attendance_Prediction.ipynb  step-by-step notebook (run it to see the results)
├── attendance_prediction_demo.html      interactive demo - open in any browser
├── data/
│   └── student_attendance_data.csv      the synthetic dataset
└── outputs/
    └── confusion_matrix.png
```

A `models/` folder is created the first time you train (trained models are not
shipped, so they are always built with the scikit-learn version on your machine).

---

## Setup (once)

You need **Python 3.9 or newer** (developed and tested on 3.12). Open a terminal
**inside the project folder**, then:

**Windows (Command Prompt or PowerShell)**

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

(The virtual environment is optional but keeps the packages tidy. If `python`
isn't found on Windows, try `py`.)

---

## Run it

### 1. Train, evaluate and save the model

```
python attendance_predictor.py
```

Runs every stage of the pipeline - preprocessing, feature selection, 80/20
train/test split, training, evaluation (accuracy, precision, recall, F1,
confusion matrix) - then saves the models to `models/`, the confusion matrix to
`outputs/`, and finishes with a demo prediction.

### 2. Predict for a student

```
python predict.py --delivered 100 --attended 62 --medical 4 --duty 2
```

Or run it with no options and it asks for each value:

```
python predict.py
```

If you haven't trained yet, `predict.py` trains a model automatically first.

### 3. Run the web app (students + teachers)

```
python app.py
```

Open http://127.0.0.1:5000. The first run creates `instance/app.db` (SQLite), a
secret key and a **teacher access code**, and prints the code in the terminal:

```
  Teacher sign-up code: TCH-7K3M-QX2P   (stored in instance/teacher_code.txt)
```

**Signing up**

- **Students** choose *Student* and enter name, email, **roll number**, **class /
  section** and a password. They appear in the teacher's class list immediately.
- **Teachers** choose *Teacher* and must enter the **teacher access code**. Without
  it a teacher account cannot be created (wrong codes are throttled). Give the
  code only to staff. A logged-in teacher can see it - and generate a new one - on
  their Profile page. To use your own fixed code, set the `TEACHER_CODE`
  environment variable instead.

| Page | Who | What it does |
|---|---|---|
| `/` | everyone | landing page |
| `/register`, `/login` | everyone | account creation, password-strength meter |
| `/dashboard` | students | attendance ring, can-skip / must-attend message, breakdown bar, prediction, trend chart, lecture planner |
| `/predict` | students | enter the four numbers, **live** preview; *Save* records it for the teacher to see |
| `/history` | students | search, risk filter, sortable columns, delete, CSV export |
| `/class` | teachers | **class overview**: totals, class average, shortage / high-risk counts, distribution chart, "needs attention" list, per-class summary, searchable and filterable table of **all** registered students (including those with no data yet), CSV export |
| `/class/student/<id>` | teachers | one student's full record: details, current numbers, prediction, trend, complete history, private teacher notes |
| `/profile` | everyone | change details / password, delete account; teachers also see the access code |

Teachers do not enter attendance for students - students enter their own, and
the class overview always shows each student's **latest saved** numbers.

Light/dark theme toggle is in the top bar. Security basics: hashed passwords,
CSRF tokens on every POST, login and teacher-code throttling, teacher-only pages
enforced on the server, spreadsheet-formula protection in exported CSVs, safe
redirects. For real deployment set `SECRET_KEY` as an environment variable and
run behind HTTPS with a production server (e.g. gunicorn).

**Upgrading an existing database:** just start the new version - it adds the
roll number and class columns to your existing users and creates the new tables.
Old accounts keep working (existing students should fill in roll number / class
on their Profile page; they will show as "-" until then). Attendance saved by the
old version used different inputs and is not carried over.

JSON API (needs a logged-in student session + `X-CSRF-Token` header):
`POST /api/what-if` (predict only) and `POST /api/predict` (predict and save),
body `{"Lectures_Delivered": 100, "Lectures_Attended": 62, "Medical_Leave": 4, "Duty_Leave": 2}`.

### 4. Predict for a whole class (CSV)

```
python batch_predict.py students.csv
```

`students.csv` needs the four columns `Lectures_Delivered`, `Lectures_Attended`, `Medical_Leave`, `Duty_Leave` (extra columns like `Name` are kept).
Results go to `students_predictions.csv`; invalid rows get an `Error` message
instead of stopping the run.

### 5. Run the tests

```
python -m pytest -q
```

### 6. Open the notebook

```
pip install notebook
jupyter notebook Student_Attendance_Prediction.ipynb
```

Start Jupyter **from the project folder** - the notebook imports
`attendance_predictor.py` from the same directory. (VS Code also opens `.ipynb`
files directly.) Run all cells to see the outputs and plots.

### 7. Open the interactive demo

Double-click `attendance_prediction_demo.html`, or:

```
start attendance_prediction_demo.html        (Windows)
open attendance_prediction_demo.html         (macOS)
xdg-open attendance_prediction_demo.html     (Linux)
```

No server needed - the demo runs the same model maths in the browser.

---

## Use your own data

Put your records in a CSV with these columns:

| Column                  | Meaning                                                    |
|-------------------------|------------------------------------------------------------|
| `Lectures_Delivered`    | lectures held so far                                       |
| `Lectures_Attended`     | lectures the student attended                              |
| `Medical_Leave`         | lectures missed on approved medical leave                  |
| `Duty_Leave`            | lectures missed on approved duty leave                     |
| `Future_Attendance`     | the student's actual attendance % in the **next** period   |
| `Sufficient_Attendance` | *(optional)* 1 if that was enough, else 0 - if left out, it is derived from `Future_Attendance` and the required % |

Rows where attended + medical + duty is more than delivered are dropped as invalid.

Then train on it and refresh the demo:

```
python attendance_predictor.py --data my_records.csv
python export_demo.py
```

`export_demo.py` copies the newly trained coefficients into the HTML demo so it
matches your model.

---

## How it works

```
STUDENT DATA -> Preprocessing -> Feature Selection -> Train/Test Split
   -> ML Model -> Model Evaluation -> Trained Model -> Prediction System
   -> Student Input -> Attendance Status / Expected % / Risk Level
```

Two models are trained on the same four (standardized) features, all derived from the
lecture counts - `Attendance_Pct`, `Medical_Pct`, `Duty_Pct`, `Lectures_Delivered`:

- **Logistic Regression** - classifies whether future attendance will be
  *sufficient*. Its probability gives the risk level: **HIGH** if the chance of
  falling short is 66% or more, **MEDIUM** from 33% to 66%, **LOW** below 33%.
  Status is *Safe* only for LOW, otherwise *At Risk*.
- **Linear Regression** - predicts the attendance percentage itself, since
  "Predicted Attendance: 68.6%" needs a number, not just a class.

The percentage, *lectures you can still miss* and *lectures you must attend in a
row* are exact arithmetic, not model output. The recommendation text is rule-based
advice built from those numbers and the model's risk level.

### Results on the bundled synthetic data

Held-out test set (240 students):

| Logistic Regression (classifier) | |
|---|---|
| Accuracy  | 0.875 |
| Precision | 0.870 |
| Recall    | 0.855 |
| F1-score  | 0.862 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[116, 14], [16, 94]]` |

Linear Regression (attendance %): MAE 4.0 points, RMSE 5.0 points, R² 0.89.

Because the data is synthetic, and because with only these inputs the best predictor
of next period's attendance is simply current attendance, these numbers show the
pipeline works - they are not a promise about your students.

Exact figures can differ slightly with other library versions.

---

## Configuration

- **Required attendance** - `REQUIRED_ATTENDANCE` at the top of
  `attendance_predictor.py` (default 75). It defines what "sufficient" means to
  the classifier, so after changing it, retrain (`python attendance_predictor.py`).
- **Risk tiers** - the 0.66 / 0.33 cut-offs in `determine_risk_level()`.
- **Recommendation rules** - `generate_recommendation()`.
- **Do leaves count as present?** - `COUNT_MEDICAL_LEAVE`, `COUNT_DUTY_LEAVE`.
- **Teacher code** - `TEACHER_CODE` environment variable, or `instance/teacher_code.txt`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| `python` not found | Use `python3` (macOS/Linux) or `py` (Windows) |
| PowerShell blocks `venv\Scripts\activate` | Use Command Prompt, or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first |
| Notebook can't import `attendance_predictor` | Start Jupyter from inside the project folder |
| Warning about a different scikit-learn version when loading a model | Delete the `models/` folder and run `python attendance_predictor.py` again |
| Status icon shows as `?` in an old Windows console | Cosmetic only - the prediction is unaffected |

## Limitations

Four inputs and a linear/logistic model can't capture everything that affects
attendance (illness, family circumstances, timetable clashes). Treat the output
as an early-warning signal, not a verdict. Students type in their own numbers, so the
teacher view is only as accurate as what they enter - check any figures that look
odd on the student's detail page.
