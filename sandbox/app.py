"""BillTrack: a small simulated internal accounts-payable system.

Deliberately imperfect so the worker has real failure modes to handle: transient 503s,
session expiry, strict input formats, duplicate-invoice rejection.
"""
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.serving import make_server

SCHEMA = """
CREATE TABLE vendors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT);
CREATE TABLE bills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_id INTEGER NOT NULL REFERENCES vendors(id),
    invoice_no TEXT NOT NULL UNIQUE,
    amount_cents INTEGER NOT NULL,
    due_date TEXT NOT NULL,
    notes TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, action TEXT NOT NULL, bill_id INTEGER, detail TEXT);
"""

AMOUNT_RE = re.compile(r"^\d{1,3}(,\d{3})*(\.\d{1,2})?$|^\d+(\.\d{1,2})?$")
DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")


def init_db(path: Path, vendors: list[tuple], bills: list[dict]) -> None:
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO vendors VALUES (?,?,?)", vendors)
    now = _now()
    for b in bills:
        db.execute(
            "INSERT INTO bills (vendor_id, invoice_no, amount_cents, due_date, notes, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (b["vendor_id"], b["invoice_no"], b["amount_cents"], b["due_date"], b.get("notes", ""),
             b.get("status", "pending"), now, now))
    db.commit()
    db.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_amount(raw: str) -> int | None:
    raw = raw.strip()
    if not AMOUNT_RE.match(raw):
        return None
    return round(float(raw.replace(",", "")) * 100)


def parse_due_date(raw: str) -> str | None:
    m = DATE_RE.match(raw.strip())
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return datetime(y, mo, d).date().isoformat()
    except ValueError:
        return None


def create_app(db_path: Path, faults: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = "sandbox-only"
    faults = dict(faults or {})
    state = {"post_failures_left": int(faults.get("fail_first_posts", 0)),
             "requests_until_expiry": faults.get("session_expires_after")}
    lock = threading.Lock()

    def db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = sqlite3.connect(db_path)
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        if "db" in g:
            g.db.close()

    @app.template_filter("money")
    def money(cents: int) -> str:
        return f"${cents / 100:,.2f}"

    @app.before_request
    def gate():
        if request.endpoint in ("login", "static", "health"):
            return
        with lock:
            n = state["requests_until_expiry"]
            if n is not None:
                if n <= 0:
                    session.clear()
                    state["requests_until_expiry"] = None
                else:
                    state["requests_until_expiry"] = n - 1
        if not session.get("user"):
            return redirect(url_for("login", next=request.full_path.rstrip("?")))

    @app.get("/health")
    def health():
        return "ok"

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            session["user"] = "sandbox-user"
            return redirect(request.form.get("next") or url_for("bills"))
        return render_template("login.html", next=request.args.get("next", ""))

    @app.get("/")
    def index():
        return redirect(url_for("bills"))

    @app.get("/vendors")
    def vendors():
        rows = db().execute("SELECT * FROM vendors ORDER BY name").fetchall()
        return render_template("vendors.html", vendors=rows)

    @app.get("/bills")
    def bills():
        q = request.args.get("q", "").strip()
        sql = "SELECT b.*, v.name AS vendor FROM bills b JOIN vendors v ON v.id = b.vendor_id"
        args: list = []
        if q:
            sql += " WHERE v.name LIKE ? OR b.invoice_no LIKE ?"
            args = [f"%{q}%", f"%{q}%"]
        rows = db().execute(sql + " ORDER BY b.id DESC", args).fetchall()
        return render_template("bills.html", bills=rows, q=q)

    @app.get("/bills/new")
    def new_bill():
        vendors_ = db().execute("SELECT * FROM vendors ORDER BY name").fetchall()
        return render_template("bill_form.html", vendors=vendors_, bill=None, form={}, error=None)

    @app.post("/bills")
    def create_bill():
        with lock:
            if state["post_failures_left"] > 0:
                state["post_failures_left"] -= 1
                return render_template("unavailable.html"), 503
        form = request.form
        vendors_ = db().execute("SELECT * FROM vendors ORDER BY name").fetchall()

        def fail(msg: str):
            return render_template("bill_form.html", vendors=vendors_, bill=None, form=form, error=msg), 400

        vendor_id = form.get("vendor_id", "")
        if not vendor_id.isdigit() or not db().execute("SELECT 1 FROM vendors WHERE id=?", (vendor_id,)).fetchone():
            return fail("Select a vendor.")
        invoice_no = form.get("invoice_no", "").strip()
        if not invoice_no:
            return fail("Invoice number is required.")
        dup = db().execute("SELECT id FROM bills WHERE invoice_no=?", (invoice_no,)).fetchone()
        if dup:
            return fail(f"Duplicate invoice number: {invoice_no} already exists as bill #{dup['id']}.")
        cents = parse_amount(form.get("amount", ""))
        if not cents:
            return fail("Amount must be a plain number such as 1234.56 (no currency symbol).")
        due = parse_due_date(form.get("due_date", ""))
        if not due:
            return fail("Due date must be in DD/MM/YYYY format.")
        cur = db().execute(
            "INSERT INTO bills (vendor_id, invoice_no, amount_cents, due_date, notes, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,'pending',?,?)",
            (vendor_id, invoice_no, cents, due, form.get("notes", "").strip(), _now(), _now()))
        db().execute("INSERT INTO audit (ts, action, bill_id, detail) VALUES (?,?,?,?)",
                     (_now(), "create", cur.lastrowid, invoice_no))
        db().commit()
        flash(f"Bill {invoice_no} saved.")
        return redirect(url_for("bill_detail", bill_id=cur.lastrowid))

    def load_bill(bill_id: int):
        row = db().execute("SELECT b.*, v.name AS vendor FROM bills b JOIN vendors v ON v.id=b.vendor_id"
                           " WHERE b.id=?", (bill_id,)).fetchone()
        if not row:
            abort(404)
        return row

    @app.get("/bills/<int:bill_id>")
    def bill_detail(bill_id: int):
        return render_template("bill_detail.html", bill=load_bill(bill_id))

    @app.get("/bills/<int:bill_id>/edit")
    def edit_bill(bill_id: int):
        return render_template("bill_form.html", vendors=[], bill=load_bill(bill_id), form={}, error=None)

    @app.post("/bills/<int:bill_id>/edit")
    def update_bill(bill_id: int):
        bill = load_bill(bill_id)
        if bill["status"] == "paid":
            return render_template("bill_form.html", vendors=[], bill=bill, form=request.form,
                                   error="Paid bills cannot be edited."), 400
        cents = parse_amount(request.form.get("amount", ""))
        due = parse_due_date(request.form.get("due_date", ""))
        if not cents:
            error = "Amount must be a plain number such as 1234.56 (no currency symbol)."
        elif not due:
            error = "Due date must be in DD/MM/YYYY format."
        else:
            error = None
        if error:
            return render_template("bill_form.html", vendors=[], bill=bill, form=request.form, error=error), 400
        db().execute("UPDATE bills SET amount_cents=?, due_date=?, notes=?, updated_at=? WHERE id=?",
                     (cents, due, request.form.get("notes", "").strip(), _now(), bill_id))
        db().execute("INSERT INTO audit (ts, action, bill_id, detail) VALUES (?,?,?,?)",
                     (_now(), "update", bill_id, f"amount={cents} due={due}"))
        db().commit()
        flash("Bill updated.")
        return redirect(url_for("bill_detail", bill_id=bill_id))

    @app.post("/bills/<int:bill_id>/pay")
    def pay_bill(bill_id: int):
        bill = load_bill(bill_id)
        if bill["status"] == "paid":
            flash("Bill is already paid.")
        else:
            db().execute("UPDATE bills SET status='paid', updated_at=? WHERE id=?", (_now(), bill_id))
            db().execute("INSERT INTO audit (ts, action, bill_id, detail) VALUES (?,?,?,?)",
                         (_now(), "pay", bill_id, str(bill["amount_cents"])))
            db().commit()
            flash(f"Payment of {bill['amount_cents'] / 100:,.2f} sent for {bill['invoice_no']}.")
        return redirect(url_for("bill_detail", bill_id=bill_id))

    return app


class SandboxServer:
    """Runs the Flask app on a background thread; port=0 picks a free one."""

    def __init__(self, app: Flask, port: int = 0):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self._server = make_server("127.0.0.1", port, app, threaded=True)
        self.port = self._server.server_port
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "SandboxServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
