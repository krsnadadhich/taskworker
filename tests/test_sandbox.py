import pytest

from sandbox.app import create_app, init_db, parse_amount, parse_due_date
from sandbox.world import BASE_BILLS, VENDORS


@pytest.fixture
def client(tmp_path):
    def make(faults=None):
        db = tmp_path / "t.db"
        init_db(db, VENDORS, BASE_BILLS)
        c = create_app(db, faults).test_client()
        c.post("/login", data={})
        return c
    return make


GOOD = {"vendor_id": "1", "invoice_no": "INV-X-1", "amount": "10.50", "due_date": "24/10/2026", "notes": ""}


def test_parsers():
    assert parse_amount("2,318.75") == 231875
    assert parse_amount("$5") is None
    assert parse_due_date("24/10/2026") == "2026-10-24"
    assert parse_due_date("2026-10-24") is None
    assert parse_due_date("31/02/2026") is None


def test_create_and_duplicate(client):
    c = client()
    assert c.post("/bills", data=GOOD).status_code == 302
    r = c.post("/bills", data=GOOD)
    assert r.status_code == 400 and b"Duplicate invoice number" in r.data


def test_validation_messages_name_the_format(client):
    c = client()
    assert b"DD/MM/YYYY" in c.post("/bills", data={**GOOD, "due_date": "2026-10-24"}).data
    assert b"no currency symbol" in c.post("/bills", data={**GOOD, "amount": "$10"}).data


def test_transient_503_then_success(client):
    c = client({"fail_first_posts": 1})
    assert c.post("/bills", data=GOOD).status_code == 503
    assert c.post("/bills", data=GOOD).status_code == 302


def test_session_expiry_redirects_to_login(client):
    c = client({"session_expires_after": 1})
    assert c.get("/bills").status_code == 200
    r = c.get("/bills")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_paid_bill_cannot_be_edited(client):
    c = client()
    r = c.post("/bills/1/edit", data={"amount": "1", "due_date": "01/01/2027"})
    assert r.status_code == 400
