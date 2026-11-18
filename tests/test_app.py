"""Web app tests. Run with: python -m pytest -q"""
import os, re, sqlite3, sys, tempfile
from pathlib import Path

os.environ["ATTENDANCE_DB"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["TEACHER_CODE"] = "TCH-TEST-2026"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as webapp  # noqa: E402

CODE = "TCH-TEST-2026"
BRIEF = {"Lectures_Delivered": 100, "Lectures_Attended": 62, "Medical_Leave": 4, "Duty_Leave": 2}
_n = iter(range(10_000))


def token(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)


def register(client, email, role="student", pw="secret123", name="Test User", roll=None, cls="CSE-3A", code=CODE):
    data = dict(csrf_token=token(client, "/register"), name=name, email=email, role=role,
                password=pw, confirm=pw, roll_no=roll or f"R{next(_n)}", class_name=cls, teacher_code=code)
    return client.post("/register", data=data, follow_redirects=True)


def student(email, name="Test User", **kw):
    c = webapp.app.test_client()
    register(c, email, name=name, **kw)
    return c


def save(c, data=BRIEF):
    return c.post("/api/predict", json=data, headers={"X-CSRF-Token": token(c, "/predict")})


def uid(email):
    return sqlite3.connect(webapp.DB_PATH).execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()[0]


def teacher(email="t@example.com"):
    c = webapp.app.test_client()
    register(c, email, role="teacher", name="Mr Teacher")
    return c


# ---------------------------------------------------------------- students
def test_full_student_flow():
    c = webapp.app.test_client()
    assert c.get("/dashboard").status_code == 302                       # protected
    r = register(c, "stu@example.com")
    assert b"No attendance recorded yet" in r.data
    tok = token(c, "/predict")
    r = c.post("/api/predict", json=BRIEF, headers={"X-CSRF-Token": tok})
    d = r.get_json()
    assert r.status_code == 200 and d["risk_level"] == "HIGH" and d["current_attendance"] == 68.0 and d["must_attend"] == 28
    assert c.post("/api/what-if", json=BRIEF, headers={"X-CSRF-Token": tok}).status_code == 200
    assert c.post("/api/predict", json=BRIEF).status_code == 400          # no CSRF token
    dash = c.get("/dashboard").data
    assert b"Attendance trend" in dash and b"68.0%" in dash and b"Lecture planner" in dash
    assert b"Attend your next <b>28</b>" in dash
    assert c.get("/history/export").data.startswith(b"Date,")
    assert c.get("/class").status_code == 403                             # students can't see class
    assert c.get("/class/student/1").status_code == 403


def test_internal_marks_are_gone_from_the_student_ui():
    c = student("nomarks@example.com")
    save(c)
    for path in ("/predict", "/dashboard", "/history"):
        html = c.get(path).get_data(as_text=True).lower()
        assert "internal" not in html and "study hours" not in html and "assignment" not in html
    html = c.get("/predict").get_data(as_text=True)
    for label in ("Lectures delivered", "Lectures attended", "Medical leave", "Duty leave"):
        assert label in html


def test_inconsistent_or_bad_numbers_rejected_by_api():
    c = student("bad@example.com")
    tok = token(c, "/predict")
    h = {"X-CSRF-Token": tok}
    assert c.post("/api/predict", json={**BRIEF, "Lectures_Attended": 99, "Medical_Leave": 5}, headers=h).status_code == 400
    assert c.post("/api/predict", json={**BRIEF, "Duty_Leave": -2}, headers=h).status_code == 400
    assert c.post("/api/predict", json={**BRIEF, "Lectures_Attended": 10.5}, headers=h).status_code == 400
    assert c.post("/api/predict", json={"Lectures_Delivered": 10}, headers=h).status_code == 400
    assert b"No attendance recorded yet" in c.get("/dashboard").data    # nothing was saved


def test_update_page_prefills_last_saved_numbers_and_history_delete():
    c = student("prefill@example.com")
    save(c, {"Lectures_Delivered": 40, "Lectures_Attended": 31, "Medical_Leave": 1, "Duty_Leave": 0})
    assert b'value="31"' in c.get("/predict").data
    html = c.get("/history").get_data(as_text=True)
    rid = re.search(r"/history/(\d+)/delete", html).group(1)
    c.post(f"/history/{rid}/delete", data={"csrf_token": token(c, "/history")})
    assert b"Nothing saved yet" in c.get("/history").data


def test_users_cannot_delete_each_others_records():
    a, b = student("a1@example.com"), student("b1@example.com")
    save(a)
    rid = re.search(r"/history/(\d+)/delete", a.get("/history").get_data(as_text=True)).group(1)
    b.post(f"/history/{rid}/delete", data={"csrf_token": token(b, "/history")})
    assert b"68.0%" in a.get("/history").data


# ---------------------------------------------------------------- sign-up rules
def test_student_signup_needs_roll_number_and_class():
    c = webapp.app.test_client()
    assert b"roll number" in register(c, "r1@example.com", roll="!!").data
    assert b"class / section" in register(c, "r2@example.com", cls="").data


def test_duplicate_roll_number_in_same_class_blocked_but_other_class_ok():
    student("d1@example.com", roll="21CS001", cls="CSE-3A")
    c = webapp.app.test_client()
    assert b"already registered in that class" in register(c, "d2@example.com", roll="21cs001", cls="cse-3a").data
    assert b"Your account is ready" in register(webapp.app.test_client(), "d3@example.com", roll="21CS001", cls="CSE-3B").data


def test_teacher_signup_requires_the_access_code():
    c = webapp.app.test_client()
    assert b"access code is not correct" in register(c, "tt1@example.com", role="teacher", code="").data
    assert b"access code is not correct" in register(c, "tt2@example.com", role="teacher", code="WRONG").data
    assert c.get("/class").status_code == 302                                # still logged out
    ok = register(webapp.app.test_client(), "tt3@example.com", role="teacher", code=CODE.lower())  # case-insensitive
    assert b"Class overview" in ok.data


def test_student_cannot_become_teacher_by_tampering():
    c = webapp.app.test_client()
    register(c, "sneaky@example.com", role="teacher", code="nope")
    assert c.get("/class").status_code == 302


def test_teacher_code_attempts_are_throttled():
    webapp._attempts.clear()
    c = webapp.app.test_client()
    for i in range(5):
        register(c, f"brute{i}@example.com", role="teacher", code=f"GUESS{i}")
    assert b"Too many wrong teacher codes" in register(c, "brute9@example.com", role="teacher", code=CODE).data
    webapp._attempts.clear()


def test_register_validation_and_duplicates():
    c = webapp.app.test_client()
    assert b"at least 8" in register(c, "a@example.com", pw="short").data
    assert b"letter and" in register(c, "b@example.com", pw="onlyletters").data
    register(c, "dup@example.com")
    c.post("/logout", data={"csrf_token": token(c, "/predict")})
    assert b"already exists" in register(c, "dup@example.com").data


def test_login_logout_and_wrong_password():
    c = webapp.app.test_client()
    register(c, "login@example.com")
    c.post("/logout", data={"csrf_token": token(c, "/predict")})
    r = c.post("/login", data=dict(csrf_token=token(c), email="login@example.com", password="nope12345"))
    assert b"Incorrect email or password" in r.data
    r = c.post("/login", data=dict(csrf_token=token(c), email="login@example.com", password="secret123"), follow_redirects=True)
    assert b"Welcome back" in r.data
    assert c.post("/login", data={"email": "x"}).status_code == 400       # CSRF enforced


# ---------------------------------------------------------------- teacher dashboard
def test_new_signup_appears_for_teacher_immediately_then_shows_attendance():
    t = teacher("tdash@example.com")
    s = student("fresh@example.com", name="Fresh Student", roll="F-001", cls="ECE-2B")
    page = t.get("/class").get_data(as_text=True)
    assert "Fresh Student" in page and "F-001" in page and "ECE-2B" in page and "No data" in page   # just signed up
    save(s, {"Lectures_Delivered": 70, "Lectures_Attended": 41, "Medical_Leave": 0, "Duty_Leave": 0})   # 58.6%
    page = t.get("/class").get_data(as_text=True)
    assert "Fresh Student" in page and "58.6%" in page and "Shortage" in page and "HIGH" in page
    # student updates later -> teacher sees the newest numbers only
    save(s, {"Lectures_Delivered": 120, "Lectures_Attended": 100, "Medical_Leave": 5, "Duty_Leave": 3})  # 90.0%
    page = t.get("/class").get_data(as_text=True)
    assert "90.0%" in page and "58.6%" not in page


def test_teacher_sees_students_only():
    t = teacher("tonly@example.com")
    student("kid@example.com", name="Kid Student")
    page = t.get("/class").get_data(as_text=True)
    assert "Kid Student" in page and "Mr Teacher</a>" not in page


def test_teacher_student_detail_shows_everything_and_is_teacher_only():
    t = teacher("tdetail@example.com")
    s = student("detail@example.com", name="Detail Kid", roll="D-9", cls="ME-1A")
    save(s)
    sid = uid("detail@example.com")
    d = t.get(f"/class/student/{sid}")
    html = d.get_data(as_text=True)
    assert d.status_code == 200
    for needle in ("Detail Kid", "detail@example.com", "D-9", "ME-1A", "Lectures delivered", "Medical leave",
                   "Duty leave", "Full history", "68.0%"):
        assert needle in html
    assert t.get("/class/student/999999").status_code == 404
    assert t.get(f"/class/student/{uid('tdetail@example.com')}").status_code == 404   # a teacher id is not a student
    assert s.get(f"/class/student/{sid}").status_code == 403


def test_teacher_notes_add_delete_and_students_cannot_use_them():
    t = teacher("tnotes@example.com")
    s = student("noted@example.com", name="Noted Kid")
    sid = uid("noted@example.com")
    t.post(f"/class/student/{sid}/notes", data=dict(csrf_token=token(t, "/class"), body="<b>Parent called</b>"))
    html = t.get(f"/class/student/{sid}").get_data(as_text=True)
    assert "&lt;b&gt;Parent called&lt;/b&gt;" in html and "<b>Parent called</b>" not in html    # escaped
    assert s.post(f"/class/student/{sid}/notes", data=dict(csrf_token=token(s, "/predict"), body="hack")).status_code == 403
    nid = re.search(r"/class/notes/(\d+)/delete", html).group(1)
    other = teacher("tnotes2@example.com")
    assert other.post(f"/class/notes/{nid}/delete", data=dict(csrf_token=token(other, "/class"))).status_code == 404
    t.post(f"/class/notes/{nid}/delete", data=dict(csrf_token=token(t, "/class")))
    assert "Parent called" not in t.get(f"/class/student/{sid}").get_data(as_text=True)


def test_class_export_csv_and_formula_injection_is_neutralised():
    t = teacher("texport@example.com")
    s = student("evil@example.com", name="=HYPERLINK(\"http://x\")")
    save(s)
    r = t.get("/class/export")
    text = r.get_data(as_text=True)
    assert r.mimetype == "text/csv" and text.startswith("Name,Email,Roll_No,Class,Lectures_Delivered")
    assert "'=HYPERLINK" in text and "\n=HYPERLINK" not in text
    assert s.get("/class/export").status_code == 403


def test_teacher_profile_shows_code_and_can_regenerate_only_when_file_managed():
    t = teacher("tcode@example.com")
    assert CODE in t.get("/profile").get_data(as_text=True)
    t.post("/profile", data=dict(csrf_token=token(t, "/profile"), action="new_code"))
    assert CODE in t.get("/profile").get_data(as_text=True)                  # env-managed: unchanged
    s = student("notcode@example.com")
    assert CODE not in s.get("/profile").get_data(as_text=True)              # students never see it
    s.post("/profile", data=dict(csrf_token=token(s, "/profile"), action="new_code"))


def test_teachers_are_redirected_away_from_student_pages():
    t = teacher("tredir@example.com")
    assert t.get("/predict").status_code == 302 and t.get("/history").status_code == 302
    assert t.post("/api/predict", json=BRIEF, headers={"X-CSRF-Token": token(t, "/class")}).status_code == 403


# ---------------------------------------------------------------- accounts + db
def test_profile_details_update_and_roll_clash():
    student("p1@example.com", roll="P-1", cls="X1")
    c = student("p2@example.com", roll="P-2", cls="X1")
    c.post("/profile", data=dict(csrf_token=token(c, "/profile"), action="name", name="New Name", roll_no="P-1", class_name="X1"))
    assert b"New Name" not in c.get("/dashboard").data                        # clash -> rejected
    c.post("/profile", data=dict(csrf_token=token(c, "/profile"), action="name", name="New Name", roll_no="P-3", class_name="X1"))
    assert b"New Name" in c.get("/profile").data


def test_password_change_and_account_delete_removes_data():
    c = student("pw@example.com"); save(c)
    c.post("/profile", data=dict(csrf_token=token(c, "/profile"), action="password", current="secret123",
                                 new="newpass456", confirm="newpass456"))
    c.post("/logout", data={"csrf_token": token(c, "/profile")})
    ok = c.post("/login", data=dict(csrf_token=token(c), email="pw@example.com", password="newpass456"), follow_redirects=True)
    assert b"Welcome back" in ok.data
    c.post("/profile", data=dict(csrf_token=token(c, "/profile"), action="delete", current="newpass456"))
    assert c.get("/dashboard").status_code == 302
    n = sqlite3.connect(webapp.DB_PATH).execute(
        "SELECT COUNT(*) FROM attendance_records WHERE user_id NOT IN (SELECT id FROM users)").fetchone()[0]
    assert n == 0


def test_open_redirect_blocked():
    c = webapp.app.test_client(); register(c, "redir@example.com")
    c.post("/logout", data={"csrf_token": token(c, "/predict")})
    r = c.post("/login?next=//evil.com", data=dict(csrf_token=token(c), email="redir@example.com", password="secret123"))
    assert "evil.com" not in r.headers["Location"]


def test_old_database_is_upgraded_without_losing_users():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'student', created_at TEXT NOT NULL);
        INSERT INTO users (name,email,password_hash,role,created_at) VALUES ('Old','old@x.com','h','student','2025-01-01');""")
    webapp.init_db(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    assert {"roll_no", "class_name"} <= cols
    assert conn.execute("SELECT name FROM users").fetchone()[0] == "Old"
    webapp.init_db(conn)                                                     # safe to run twice


def test_legacy_student_without_roll_or_class_still_works():
    """Accounts created by the previous version have no roll number / class."""
    c = student("legacy@example.com")
    with sqlite3.connect(webapp.DB_PATH) as conn:
        conn.execute("UPDATE users SET roll_no = NULL, class_name = NULL WHERE email = 'legacy@example.com'")
    for path in ("/dashboard", "/profile", "/predict"):
        r = c.get(path)
        assert r.status_code == 200 and b"None" not in r.data
    t = teacher("tlegacy@example.com")
    assert t.get("/class").status_code == 200
    assert t.get(f"/class/student/{uid('legacy@example.com')}").status_code == 200
    assert t.get("/class/export").status_code == 200
