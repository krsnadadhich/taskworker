"""Real browser + real sandbox, scripted model: exercises snapshots, the approval gate and failure signals."""
import re

import pytest

from scenarios.catalog import ACME_FILES
from sandbox.world import build_world
from worker.human import ScriptedHuman
from worker.llm.base import ModelTurn, ResultsMsg, ToolCall
from worker.llm.scripted import ScriptedLLM, turn
from worker.policy import Policy
from worker.runner import run_task


def el(label_pattern):
    """Scripted step that finds an element number in the latest snapshot by its label and builds the call."""
    def build(name, **extra):
        def step(messages):
            text = [r.content for m in messages if isinstance(m, ResultsMsg) for r in m.results
                    if "interactive elements" in r.content][-1]
            m = re.search(rf"\[(\d+)\] [^\n]*\"{label_pattern}[^\"]*\"", text)
            assert m, f"no element matching {label_pattern!r} in:\n{text}"
            return ModelTurn("", [ToolCall("x", name, {"element": int(m.group(1)), **extra})])
        return step
    return build


@pytest.fixture
def world(tmp_path):
    w = build_world(tmp_path / "ws", ACME_FILES, {"fail_first_posts": 1})
    yield w
    w.close()


def test_save_bill_recovers_from_503_and_is_verified(world, tmp_path):
    script = [
        turn("browser_open", url=world.url + "/bills/new"),
        el("Sign in")("browser_click"),
        el("Vendor(?=\")")("browser_select", option="Acme Supplies Ltd"),
        el("Invoice number")("browser_type", text="INV-AC-0455"),
        el("Amount")("browser_type", text="2318.75"),
        el("Due date")("browser_type", text="24/10/2026"),
        el("Save bill")("browser_click"),          # server answers 503, nothing saved
        el("Back to form")("browser_click"),
        el("Vendor(?=\")")("browser_select", option="Acme Supplies Ltd"),
        el("Invoice number")("browser_type", text="INV-AC-0455"),
        el("Amount")("browser_type", text="2318.75"),
        el("Due date")("browser_type", text="24/10/2026"),
        el("Save bill")("browser_click"),
        turn("finish", status="done", summary="saved", evidence=["INV-AC-0455"]),
        turn("verdict", achieved=True, reasoning="checked"),
    ]
    result = run_task("enter invoice", ScriptedLLM(script), ScriptedHuman(), Policy("risky"), world.url,
                      world.root, tmp_path / "run", echo=False)
    rows = world.query("SELECT * FROM bills WHERE invoice_no='INV-AC-0455'")
    assert result.status == "done" and result.verified
    assert len(rows) == 1 and rows[0]["amount_cents"] == 231875 and rows[0]["due_date"] == "2026-10-24"
    trace = (tmp_path / "run" / "trace.jsonl").read_text()
    assert "HTTP 503" in trace or "returned HTTP 503" in trace
    assert list((tmp_path / "run" / "evidence").glob("*after_submit.png"))


def test_pay_button_requires_approval_and_denial_blocks_it(world, tmp_path):
    human = ScriptedHuman(approve=False)
    script = [
        turn("browser_open", url=world.url + "/bills/2"),
        el("Sign in")("browser_click"),
        el("Pay bill")("browser_click"),
        turn("finish", status="blocked", summary="human declined payment"),
    ]
    result = run_task("pay it", ScriptedLLM(script), human, Policy("risky"), world.url, world.root,
                      tmp_path / "run", echo=False)
    assert human.approval_requests and "Pay bill" in human.approval_requests[0]
    assert world.query("SELECT status FROM bills WHERE id=2")[0]["status"] == "pending"
    assert result.status == "blocked"


def test_browser_refuses_hosts_outside_allowlist(world, tmp_path):
    script = [turn("browser_open", url="http://example.com/"), turn("finish", status="blocked", summary="x")]
    run_task("x", ScriptedLLM(script), ScriptedHuman(), Policy("risky"), world.url, world.root,
             tmp_path / "run", echo=False)
    assert "Blocked" in (tmp_path / "run" / "trace.jsonl").read_text()
