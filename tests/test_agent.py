"""Agent-loop behaviour with fake tools and a scripted model: no browser, no network."""
import pytest

from worker.agent import Agent, Config
from worker.context import compact
from worker.human import NoHuman, ScriptedHuman
from worker.llm.base import ModelTurn, ResultsMsg, ToolCall, ToolResult, UserMsg
from worker.llm.scripted import ScriptedLLM, turn
from worker.policy import Policy
from worker.tools.base import Intent, Tool, ToolOutput, schema
from worker.trace import Trace


def make_agent(tmp_path, script, tools=None, human=None, policy=None, **cfg):
    log = []

    def read(args):
        return ToolOutput("page: hello", ephemeral=True)

    def write(args):
        log.append(args)
        return ToolOutput("saved")

    def flaky(args):
        return ToolOutput("boom", ok=False)

    default = [
        Tool("read", "r", schema({}), read, read_only=True),
        Tool("write", "w", schema({"x": {"type": "string"}}, ["x"]), write,
             preflight=lambda a: Intent("form_submit", f"write {a['x']}", {"label": "Pay bill", "fields": {}})),
        Tool("flaky", "f", schema({}), flaky),
    ]
    llm = ScriptedLLM(script)
    agent = Agent(llm, tools or default, human or NoHuman(), policy or Policy("risky"),
                  Trace(tmp_path, echo=False), "env", "2026-10-04", Config(**cfg))
    return agent, llm, log


def verdict(achieved, why="ok"):
    return turn("verdict", achieved=achieved, reasoning=why)


def test_happy_path_is_verified(tmp_path):
    agent, _, _ = make_agent(tmp_path, [
        turn("read"),
        turn("finish", status="done", summary="did it", evidence=["bill 1"]),
        verdict(True),
    ])
    r = agent.run("goal")
    assert r.status == "done" and r.verified and (tmp_path / "report.md").exists()


def test_verifier_rejection_sends_worker_back_then_accepts(tmp_path):
    agent, _, _ = make_agent(tmp_path, [
        turn("finish", status="done", summary="did it"), verdict(False, "amount differs"),
        turn("read"), turn("finish", status="done", summary="fixed"), verdict(True),
    ])
    r = agent.run("goal")
    assert r.status == "done" and r.verified and r.steps == 3


def test_repeated_verifier_rejection_ends_unverified(tmp_path):
    script = []
    for _ in range(3):
        script += [turn("finish", status="done", summary="did it"), verdict(False, "nope")]
    r = make_agent(tmp_path, script)[0].run("goal")
    assert r.status == "unverified" and not r.verified


def test_risky_action_denied_is_not_executed(tmp_path):
    human = ScriptedHuman(approve=False)
    agent, _, log = make_agent(tmp_path, [
        turn("write", x="1"),
        turn("finish", status="blocked", summary="human declined"),
    ], human=human)
    r = agent.run("goal")
    assert log == [] and r.status == "blocked" and human.approval_requests == ["write 1"]


def test_risky_action_approved_runs(tmp_path):
    agent, _, log = make_agent(tmp_path, [turn("write", x="1"), turn("finish", status="blocked", summary="s")],
                               human=ScriptedHuman(approve=True))
    agent.run("goal")
    assert log == [{"x": "1"}]


def test_unknown_tool_and_missing_args_are_reported_not_fatal(tmp_path):
    agent, llm, _ = make_agent(tmp_path, [
        turn("nope"), turn("write"), turn("finish", status="failed", summary="gave up")])
    agent.run("goal")
    results = [m for m in llm.seen[-1] if isinstance(m, ResultsMsg)]
    assert "Unknown tool" in results[0].results[0].content
    assert "Missing required" in results[1].results[0].content


def test_consecutive_errors_abort(tmp_path):
    agent, _, _ = make_agent(tmp_path, [turn("flaky")] * 10, max_consecutive_errors=3)
    r = agent.run("goal")
    assert r.status == "failed" and "repeated consecutive errors" in r.summary


def test_identical_call_loop_triggers_alert(tmp_path):
    agent, llm, _ = make_agent(tmp_path, [turn("read")] * 3 + [turn("finish", status="blocked", summary="x")])
    agent.run("goal")
    last_note = [m for m in llm.seen[-1] if isinstance(m, ResultsMsg)][-1].note
    assert "identical call 3 times" in last_note


def test_step_budget(tmp_path):
    agent, _, _ = make_agent(tmp_path, [turn("read")] * 10, max_steps=3)
    assert agent.run("goal").status == "out_of_steps"


def test_ask_user_without_human_tells_model_not_to_guess(tmp_path):
    agent, llm, _ = make_agent(tmp_path, [turn("ask_user", question="which?"),
                                          turn("finish", status="needs_input", summary="need vendor")])
    r = agent.run("goal")
    assert r.status == "needs_input"
    assert "Do not guess" in llm.seen[-1][-1].results[0].content


def test_extra_parallel_calls_are_skipped(tmp_path):
    two = ModelTurn("", [ToolCall("a", "write", {"x": "1"}), ToolCall("b", "write", {"x": "2"})])
    agent, _, log = make_agent(tmp_path, [two, turn("finish", status="blocked", summary="s")],
                               human=ScriptedHuman(approve=True))
    agent.run("goal")
    assert log == [{"x": "1"}]


def test_memory_and_plan_appear_in_state_note(tmp_path):
    agent, llm, _ = make_agent(tmp_path, [
        turn("plan", steps=[{"step": "look", "status": "doing"}]),
        turn("remember", key="amount", value="12.00"),
        turn("finish", status="blocked", summary="s")])
    agent.run("goal")
    note = [m for m in llm.seen[-1] if isinstance(m, ResultsMsg)][-1].note
    assert "amount = 12.00" in note


def test_compact_elides_only_stale_snapshots():
    snap = lambda i: ResultsMsg([ToolResult(str(i), "browser_read", f"Opened.\nURL: u{i}\nTitle: t\nbody\nmore", ephemeral=True)])
    keep = ResultsMsg([ToolResult("k", "read_file", "important file text")])
    msgs = [UserMsg("g"), snap(1), keep, snap(2)]
    out = compact(msgs, keep=1)
    assert "elided" in out[1].results[0].content and "body" not in out[1].results[0].content
    assert out[2].results[0].content == "important file text"
    assert "body" in out[3].results[0].content


def test_verifier_is_told_what_the_user_answered(tmp_path):
    human = ScriptedHuman(lambda q, o: "Northwind Traders")
    agent, llm, _ = make_agent(tmp_path, [
        turn("ask_user", question="which Northwind?"),
        turn("finish", status="done", summary="entered"),
        verdict(True),
    ], human=human)
    agent.run("goal")
    prompt = llm.seen[-1][0].text
    assert "which Northwind? -> user answered: Northwind Traders" in prompt


def test_rejection_message_tells_worker_not_to_create_extra_records(tmp_path):
    agent, llm, _ = make_agent(tmp_path, [
        turn("finish", status="done", summary="x"), verdict(False, "nope"),
        turn("finish", status="blocked", summary="y")])
    agent.run("goal")
    msg = [m for m in llm.seen[-1] if isinstance(m, ResultsMsg)][-1].results[0].content
    assert "never create extra records" in msg
