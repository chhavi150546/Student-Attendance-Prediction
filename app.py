"""
AttendIQ - web app for the Student Attendance Prediction System.

    python app.py            then open http://127.0.0.1:5000

Students:  /dashboard  /predict (update attendance)  /history  /profile
Teachers:  /class (overview)  /class/student/<id> (full detail)  /class/export  /profile
Everyone:  /  /register  /login

Teacher sign-up needs the teacher access code: set the TEACHER_CODE environment
variable, or use the one generated in instance/teacher_code.txt (printed when
the app starts).
Data:   SQLite file at instance/app.db (created automatically)
"""

import csv
import io
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (Flask, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from attendance_predictor import (FIELD_LIMITS, INPUT_FIELDS, REQUIRED_ATTENDANCE,
                                  attendance_stats, load_trained_system, predict_attendance,
                                  run_pipeline)

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)
DB_PATH = Path(os.environ.get("ATTENDANCE_DB", INSTANCE_DIR / "app.db"))


def load_secret_key() -> str:
    """SECRET_KEY env var, else a random key stored once in instance/secret.key."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    key_file = INSTANCE_DIR / "secret.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    return key_file.read_text().strip()


CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # no 0/O/1/I to avoid typos
CODE_FILE = INSTANCE_DIR / "teacher_code.txt"
CODE_FROM_ENV = bool(os.environ.get("TEACHER_CODE", "").strip())


def new_teacher_code() -> str:
    pick = lambda n: "".join(secrets.choice(CODE_ALPHABET) for _ in range(n))
    return f"TCH-{pick(4)}-{pick(4)}"


def current_teacher_code() -> str:
    """TEACHER_CODE env var, else the code stored in instance/teacher_code.txt
    (created on first use). Read on every call so a regenerated code works at once."""
    if CODE_FROM_ENV:
        return os.environ["TEACHER_CODE"].strip()
    if not CODE_FILE.exists():
        CODE_FILE.write_text(new_teacher_code())
    return CODE_FILE.read_text().strip()


app = Flask(__name__)
app.config.update(SECRET_KEY=load_secret_key(), SESSION_COOKIE_HTTPONLY=True,
                  SESSION_COOKIE_SAMESITE="Lax", MAX_CONTENT_LENGTH=1024 * 1024)

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def get_models():
    try:
        return load_trained_system()
    except FileNotFoundError:
        import matplotlib
        matplotlib.use("Agg")
        run_pipeline()
        return load_trained_system()


CLF, REG, SCALER = get_models()

FORM_FIELDS = [  # key, label, hint, default
    ("Lectures_Delivered", "Lectures delivered", "total lectures held so far", 100),
    ("Lectures_Attended",  "Lectures attended",  "classes you were present for", 62),
    ("Medical_Leave",      "Medical leave",      "lectures missed on approved medical leave", 4),
    ("Duty_Leave",         "Duty leave",         "lectures missed on approved duty leave", 2),
]


def run_prediction(data: dict) -> dict:
    result = predict_attendance(data, CLF, REG, SCALER)
    result["status"] = "Safe" if result["risk_level"] == "LOW" else "At Risk"
    return result


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'student',
    roll_no TEXT,
    class_name TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delivered INTEGER NOT NULL, attended INTEGER NOT NULL,
    medical INTEGER NOT NULL, duty INTEGER NOT NULL, absent INTEGER NOT NULL,
    attendance_pct REAL NOT NULL,
    predicted_attendance REAL, risk_level TEXT, risk_probability REAL,
    status TEXT, recommendation TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teacher_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    teacher_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_user ON attendance_records(user_id, id);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Creates tables and upgrades a database made by an older version
    (adds the roll_no / class_name columns; the old `predictions` table is left untouched)."""
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    for col in ("roll_no", "class_name"):
        if col not in have:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
    conn.commit()


def db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_):
    conn = g.pop("db", None)
    if conn:
        conn.close()


with sqlite3.connect(DB_PATH) as _c:
    init_db(_c)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# Auth helpers: sessions, CSRF, decorators, login throttling
# --------------------------------------------------------------------------
@app.before_request
def load_user():
    g.user = None
    uid = session.get("user_id")
    if uid:
        g.user = db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        if g.user is None:
            session.clear()


def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.before_request
def check_csrf():
    if request.method == "POST":
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
            if request.path.startswith("/api/"):
                return jsonify(error="Session expired - refresh the page."), 400
            abort(400, "Invalid or missing security token. Go back, refresh and try again.")


@app.context_processor
def inject_globals():
    return dict(csrf_token=csrf_token, user=g.get("user"), required=REQUIRED_ATTENDANCE)


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if g.user is None:
            if request.path.startswith("/api/"):
                return jsonify(error="Please log in."), 401
            flash("Please log in to continue.", "info")
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapped


def teacher_required(view):
    @wraps(view)
    @login_required
    def wrapped(*a, **kw):
        if g.user["role"] != "teacher":
            abort(403)
        return view(*a, **kw)
    return wrapped


def student_required(view):
    """Student-only pages. Teachers are sent to their class overview instead."""
    @wraps(view)
    @login_required
    def wrapped(*a, **kw):
        if g.user["role"] != "student":
            if request.path.startswith("/api/"):
                return jsonify(error="This is for student accounts."), 403
            return redirect(url_for("class_overview"))
        return view(*a, **kw)
    return wrapped


_attempts: dict = {}   # (ip, email) -> [timestamps]; in-memory, resets on restart


def too_many_attempts(key) -> bool:
    recent = [t for t in _attempts.get(key, []) if time.time() - t < 300]
    _attempts[key] = recent
    return len(recent) >= 5


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def password_problem(pw: str):
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if not (re.search(r"[A-Za-z]", pw) and re.search(r"\d", pw)):
        return "Password must contain at least one letter and one number."
    return None


def safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else None


# --------------------------------------------------------------------------
# Public + auth pages
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


ROLL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ./_-]{0,29}$")


def tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def roll_taken(class_name: str, roll_no: str, exclude_id=None) -> bool:
    row = db().execute(
        "SELECT id FROM users WHERE role='student' AND class_name = ? AND roll_no = ? AND id != ?",
        (class_name, roll_no, exclude_id or 0)).fetchone()
    return row is not None


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("dashboard"))
    form = {"name": "", "email": "", "role": "student", "roll_no": "", "class_name": ""}
    if request.method == "POST":
        form = {k: tidy(request.form.get(k, "")) for k in form}
        form["email"] = form["email"].lower()
        form["roll_no"], form["class_name"] = form["roll_no"].upper(), form["class_name"].upper()
        pw, pw2 = request.form.get("password", ""), request.form.get("confirm", "")
        code = request.form.get("teacher_code", "").strip()
        code_key = ("teacher-code", request.remote_addr)
        error = None
        if len(form["name"]) < 2:
            error = "Please enter your full name."
        elif not EMAIL_RE.match(form["email"]):
            error = "Please enter a valid email address."
        elif form["role"] not in ("student", "teacher"):
            error = "Choose a valid role."
        elif form["role"] == "student" and not ROLL_RE.match(form["roll_no"]):
            error = "Please enter your roll number (letters, numbers, - / . only)."
        elif form["role"] == "student" and not 1 <= len(form["class_name"]) <= 40:
            error = "Please enter your class / section, e.g. CSE-3A."
        elif form["role"] == "student" and roll_taken(form["class_name"], form["roll_no"]):
            error = "A student with this roll number is already registered in that class."
        elif form["role"] == "teacher" and too_many_attempts(code_key):
            error = "Too many wrong teacher codes. Please wait a few minutes and try again."
        elif form["role"] == "teacher" and not secrets.compare_digest(
                code.upper().encode(), current_teacher_code().upper().encode()):
            _attempts.setdefault(code_key, []).append(time.time())
            error = "That teacher access code is not correct. Ask your institution for the code."
        elif password_problem(pw):
            error = password_problem(pw)
        elif pw != pw2:
            error = "Passwords do not match."
        elif db().execute("SELECT 1 FROM users WHERE email = ?", (form["email"],)).fetchone():
            error = "An account with this email already exists."
        if error:
            flash(error, "error")
        else:
            is_student = form["role"] == "student"
            cur = db().execute(
                "INSERT INTO users (name, email, password_hash, role, roll_no, class_name, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (form["name"], form["email"], generate_password_hash(pw), form["role"],
                 form["roll_no"] if is_student else None, form["class_name"] if is_student else None, now()))
            db().commit()
            session.clear()
            session["user_id"] = cur.lastrowid
            flash(f"Welcome, {form['name']}! Your account is ready.", "success")
            return redirect(url_for("dashboard"))
    return render_template("register.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    email = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        key = (request.remote_addr, email)
        if too_many_attempts(key):
            flash("Too many failed attempts. Please wait a few minutes and try again.", "error")
        else:
            row = db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if row and check_password_hash(row["password_hash"], pw):
                _attempts.pop(key, None)
                session.clear()
                session["user_id"] = row["id"]
                flash(f"Welcome back, {row['name']}!", "success")
                return redirect(safe_next(request.args.get("next")) or url_for("dashboard"))
            _attempts.setdefault(key, []).append(time.time())
            flash("Incorrect email or password.", "error")
    return render_template("login.html", email=email)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def record_rows(user_id, limit=None):
    sql = "SELECT * FROM attendance_records WHERE user_id = ? ORDER BY id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return db().execute(sql, (user_id,)).fetchall()


def stats_for(rec) -> dict:
    """Exact lecture arithmetic (percentage, can-skip, must-attend...) for a saved record."""
    return attendance_stats(rec["delivered"], rec["attended"], rec["medical"], rec["duty"])


def trend_points(rows):
    """Oldest-first points for the trend chart: actual % and the model's predicted %."""
    return [{"x": r["created_at"], "y": r["attendance_pct"], "p": r["predicted_attendance"]}
            for r in reversed(rows[:20])]


def csv_safe(value):
    """Stops spreadsheet formula injection: a name like '=HYPERLINK(...)' would run in Excel."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def csv_response(rows, filename):
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


# --------------------------------------------------------------------------
# Student pages
# --------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    if g.user["role"] == "teacher":
        return redirect(url_for("class_overview"))
    rows = record_rows(g.user["id"])
    latest = rows[0] if rows else None
    return render_template("dashboard.html", latest=latest, stats=stats_for(latest) if latest else None,
                           recent=rows[:5], total=len(rows), trend=trend_points(rows))


@app.route("/predict")
@student_required
def predict_page():
    latest = record_rows(g.user["id"], 1)
    start = {"Lectures_Delivered": latest[0]["delivered"], "Lectures_Attended": latest[0]["attended"],
             "Medical_Leave": latest[0]["medical"], "Duty_Leave": latest[0]["duty"]} if latest else None
    return render_template("predict.html", fields=FORM_FIELDS, limits=FIELD_LIMITS, start=start)


@app.route("/history")
@student_required
def history():
    return render_template("history.html", rows=record_rows(g.user["id"]))


@app.route("/history/<int:rid>/delete", methods=["POST"])
@student_required
def delete_record(rid):
    db().execute("DELETE FROM attendance_records WHERE id = ? AND user_id = ?", (rid, g.user["id"]))
    db().commit()
    flash("Record deleted.", "info")
    return redirect(url_for("history"))


HISTORY_HEADER = ["Date", "Lectures_Delivered", "Lectures_Attended", "Medical_Leave", "Duty_Leave",
                  "Absent", "Attendance_Pct", "Predicted_Attendance", "Risk_Level", "Risk_Probability_Pct"]


@app.route("/history/export")
@student_required
def export_history():
    rows = [HISTORY_HEADER] + [
        [r["created_at"], r["delivered"], r["attended"], r["medical"], r["duty"], r["absent"],
         r["attendance_pct"], r["predicted_attendance"], r["risk_level"], r["risk_probability"]]
        for r in reversed(record_rows(g.user["id"]))]
    return csv_response(rows, "my_attendance.csv")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    is_student = g.user["role"] == "student"
    if request.method == "POST":
        action = request.form.get("action")
        if action == "name":
            name = tidy(request.form.get("name", ""))
            roll, cls = tidy(request.form.get("roll_no", "")).upper(), tidy(request.form.get("class_name", "")).upper()
            if len(name) < 2:
                flash("Please enter a valid name.", "error")
            elif is_student and not ROLL_RE.match(roll):
                flash("Please enter a valid roll number.", "error")
            elif is_student and not 1 <= len(cls) <= 40:
                flash("Please enter your class / section.", "error")
            elif is_student and roll_taken(cls, roll, g.user["id"]):
                flash("Another student already has that roll number in this class.", "error")
            else:
                if is_student:
                    db().execute("UPDATE users SET name = ?, roll_no = ?, class_name = ? WHERE id = ?",
                                 (name, roll, cls, g.user["id"]))
                else:
                    db().execute("UPDATE users SET name = ? WHERE id = ?", (name, g.user["id"]))
                db().commit()
                flash("Details updated.", "success")
        elif action == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not check_password_hash(g.user["password_hash"], cur):
                flash("Current password is incorrect.", "error")
            elif password_problem(new):
                flash(password_problem(new), "error")
            elif new != conf:
                flash("New passwords do not match.", "error")
            else:
                db().execute("UPDATE users SET password_hash = ? WHERE id = ?",
                             (generate_password_hash(new), g.user["id"]))
                db().commit()
                flash("Password changed.", "success")
        elif action == "new_code" and g.user["role"] == "teacher":
            if CODE_FROM_ENV:
                flash("The teacher code is set by the TEACHER_CODE environment variable - change it there.", "error")
            else:
                CODE_FILE.write_text(new_teacher_code())
                flash("New teacher code generated. The old one no longer works.", "success")
        elif action == "delete":
            if check_password_hash(g.user["password_hash"], request.form.get("current", "")):
                db().execute("DELETE FROM users WHERE id = ?", (g.user["id"],))
                db().commit()
                session.clear()
                flash("Your account and data were deleted.", "info")
                return redirect(url_for("home"))
            flash("Password is incorrect - account not deleted.", "error")
        return redirect(url_for("profile"))
    n = db().execute("SELECT COUNT(*) FROM attendance_records WHERE user_id = ?", (g.user["id"],)).fetchone()[0]
    return render_template("profile.html", n=n,
                           teacher_code=current_teacher_code() if g.user["role"] == "teacher" else None,
                           code_from_env=CODE_FROM_ENV)


# --------------------------------------------------------------------------
# Teacher pages
# --------------------------------------------------------------------------
BANDS = [("Below 50%", 0, 50), ("50 - 64%", 50, 65), ("65 - 74%", 65, 75),
         ("75 - 84%", 75, 85), ("85% +", 85, 101)]


def class_rows():
    """Every registered student with their latest record (if any), as plain dicts.
    LEFT JOIN: a student who has just signed up shows up straight away with no data."""
    rows = db().execute("""
        SELECT u.id, u.name, u.email, u.roll_no, u.class_name, u.created_at AS joined,
               r.delivered, r.attended, r.medical, r.duty, r.absent, r.attendance_pct,
               r.predicted_attendance, r.risk_level, r.risk_probability, r.recommendation,
               r.created_at AS updated
        FROM users u
        LEFT JOIN attendance_records r
               ON r.id = (SELECT MAX(id) FROM attendance_records WHERE user_id = u.id)
        WHERE u.role = 'student'""").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["has_data"] = d["delivered"] is not None
        d["stats"] = attendance_stats(d["delivered"], d["attended"], d["medical"], d["duty"]) if d["has_data"] else None
        d["state"] = "none" if not d["has_data"] else ("short" if d["stats"]["shortage"] else "ok")
        out.append(d)
    # no-data last; then highest risk first, then lowest attendance
    out.sort(key=lambda d: (not d["has_data"], -(d["risk_probability"] or 0), d["attendance_pct"] or 0,
                            d["class_name"] or "", d["roll_no"] or ""))
    return out


@app.route("/class")
@teacher_required
def class_overview():
    rows = class_rows()
    with_data = [r for r in rows if r["has_data"]]
    avg = round(sum(r["attendance_pct"] for r in with_data) / len(with_data), 1) if with_data else None
    counts = {k: sum(1 for r in with_data if r["risk_level"] == k) for k in ("HIGH", "MEDIUM", "LOW")}
    bands = [{"label": label, "n": sum(1 for r in with_data if lo <= r["attendance_pct"] < hi)}
             for label, lo, hi in BANDS]
    by_class = {}
    for r in rows:
        c = by_class.setdefault(r["class_name"] or "-", {"name": r["class_name"] or "-", "n": 0, "data": 0,
                                                         "sum": 0.0, "short": 0})
        c["n"] += 1
        if r["has_data"]:
            c["data"] += 1
            c["sum"] += r["attendance_pct"]
            c["short"] += r["state"] == "short"
    classes = sorted(by_class.values(), key=lambda c: c["name"])
    for c in classes:
        c["avg"] = round(c["sum"] / c["data"], 1) if c["data"] else None
    attention = sorted((r for r in with_data if r["state"] == "short" or r["risk_level"] == "HIGH"),
                       key=lambda r: r["attendance_pct"])[:5]
    return render_template("class.html", rows=rows, avg=avg, counts=counts, bands=bands,
                           max_band=max([b["n"] for b in bands] + [1]), classes=classes,
                           attention=attention, total=len(rows),
                           shortage=sum(r["state"] == "short" for r in rows),
                           no_data=len(rows) - len(with_data))


def get_student_or_404(sid):
    student = db().execute("SELECT * FROM users WHERE id = ? AND role = 'student'", (sid,)).fetchone()
    if student is None:
        abort(404)
    return student


@app.route("/class/student/<int:sid>")
@teacher_required
def student_detail(sid):
    student = get_student_or_404(sid)
    rows = record_rows(sid)
    notes = db().execute("""SELECT n.*, t.name AS teacher_name FROM teacher_notes n
                            JOIN users t ON t.id = n.teacher_id
                            WHERE n.student_id = ? ORDER BY n.id DESC""", (sid,)).fetchall()
    return render_template("student_detail.html", s=student, rows=rows, latest=rows[0] if rows else None,
                           stats=stats_for(rows[0]) if rows else None, trend=trend_points(rows), notes=notes)


@app.route("/class/student/<int:sid>/notes", methods=["POST"])
@teacher_required
def add_note(sid):
    get_student_or_404(sid)
    body = request.form.get("body", "").strip()
    if not 1 <= len(body) <= 500:
        flash("A note must be between 1 and 500 characters.", "error")
    else:
        db().execute("INSERT INTO teacher_notes (student_id, teacher_id, body, created_at) VALUES (?,?,?,?)",
                     (sid, g.user["id"], body, now()))
        db().commit()
        flash("Note added.", "success")
    return redirect(url_for("student_detail", sid=sid))


@app.route("/class/notes/<int:nid>/delete", methods=["POST"])
@teacher_required
def delete_note(nid):
    note = db().execute("SELECT * FROM teacher_notes WHERE id = ? AND teacher_id = ?",
                        (nid, g.user["id"])).fetchone()
    if note is None:
        abort(404)
    db().execute("DELETE FROM teacher_notes WHERE id = ?", (nid,))
    db().commit()
    flash("Note deleted.", "info")
    return redirect(url_for("student_detail", sid=note["student_id"]))


CLASS_HEADER = ["Name", "Email", "Roll_No", "Class", "Lectures_Delivered", "Lectures_Attended", "Medical_Leave",
                "Duty_Leave", "Absent", "Attendance_Pct", "Status", "Predicted_Attendance", "Risk_Level",
                "Last_Updated"]


@app.route("/class/export")
@teacher_required
def export_class():
    label = {"none": "No data", "short": "Shortage", "ok": "OK"}
    rows = [CLASS_HEADER] + [
        [csv_safe(r["name"]), csv_safe(r["email"]), csv_safe(r["roll_no"]), csv_safe(r["class_name"]),
         r["delivered"], r["attended"], r["medical"], r["duty"], r["absent"], r["attendance_pct"],
         label[r["state"]], r["predicted_attendance"], r["risk_level"], r["updated"]]
        for r in class_rows()]
    return csv_response(rows, "class_attendance.csv")


# --------------------------------------------------------------------------
# JSON API (used by the live update page)
# --------------------------------------------------------------------------
def _predict_from_request():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify(error="Send a JSON object with the four lecture counts."), 400)
    try:
        return run_prediction(data), None
    except ValueError as err:
        return None, (jsonify(error=str(err)), 400)


@app.route("/api/what-if", methods=["POST"])
@student_required
def api_what_if():
    """Prediction only - nothing is saved. Powers the live preview."""
    result, err = _predict_from_request()
    return err or jsonify(result)


@app.route("/api/predict", methods=["POST"])
@student_required
def api_predict():
    """Prediction that is saved as a new attendance record."""
    result, err = _predict_from_request()
    if err:
        return err
    cur = db().execute(
        """INSERT INTO attendance_records (user_id, delivered, attended, medical, duty, absent, attendance_pct,
           predicted_attendance, risk_level, risk_probability, status, recommendation, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (g.user["id"], result["delivered"], result["attended"], result["medical"], result["duty"],
         result["absent"], result["current_attendance"], result["predicted_attendance"], result["risk_level"],
         result["risk_probability_pct"], result["status"], result["recommendation"], now()))
    db().commit()
    result["id"] = cur.lastrowid
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify(status="ok", required_attendance=REQUIRED_ATTENDANCE)


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, message="This page is for teacher accounts only."), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="That page doesn't exist."), 404


@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", code=400, message=getattr(e, "description", "Bad request.")), 400


if __name__ == "__main__":
    print("\n  Teacher sign-up code: " + current_teacher_code()
          + ("   (from TEACHER_CODE)" if CODE_FROM_ENV else f"   (stored in {CODE_FILE.relative_to(BASE_DIR)})") + "\n")
    app.run(debug=False)
