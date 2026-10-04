"""Builds a fresh, seeded sandbox: database, inbox files and a running server."""
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from fpdf import FPDF

from .app import SandboxServer, create_app, init_db

VENDORS = [
    (1, "Acme Supplies Ltd", "billing@acme-supplies.example"),
    (2, "Northwind Traders", "ar@northwind-traders.example"),
    (3, "Northwind Logistics", "ar@northwind-logistics.example"),
    (4, "Globex Corporation", "invoices@globex.example"),
    (5, "Initech Services", "accounts@initech.example"),
]

BASE_BILLS = [
    dict(vendor_id=1, invoice_no="INV-AC-0412", amount_cents=84550, due_date="2026-09-02", status="paid"),
    dict(vendor_id=4, invoice_no="INV-G-1001", amount_cents=120000, due_date="2026-09-30"),
    dict(vendor_id=5, invoice_no="INV-IN-778", amount_cents=1540000, due_date="2026-10-20"),
]


@dataclass
class InboxFile:
    name: str
    lines: list[str]


@dataclass
class World:
    root: Path
    db_path: Path
    inbox: Path
    server: SandboxServer
    url: str
    extra: dict = field(default_factory=dict)

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    def close(self) -> None:
        self.server.stop()


def _write_inbox_file(inbox: Path, f: InboxFile) -> None:
    path = inbox / f.name
    if path.suffix.lower() == ".pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        for line in f.lines:
            pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(path))
    else:
        path.write_text("\n".join(f.lines), encoding="utf-8")


def build_world(root: Path, inbox_files: list[InboxFile], faults: dict | None = None,
                extra_bills: list[dict] | None = None, port: int = 0) -> World:
    root.mkdir(parents=True, exist_ok=True)
    inbox = root / "inbox"
    inbox.mkdir(exist_ok=True)
    for old in inbox.iterdir():
        old.unlink()
    for f in inbox_files:
        _write_inbox_file(inbox, f)
    db_path = root / "billtrack.db"
    init_db(db_path, VENDORS, BASE_BILLS + (extra_bills or []))
    server = SandboxServer(create_app(db_path, faults), port=port).start()
    return World(root=root, db_path=db_path, inbox=inbox, server=server, url=server.url)
