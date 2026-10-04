"""Command line: run a free-form task, run the eval scenarios, or just start the sandbox."""
import argparse
import json
import time
from pathlib import Path

from sandbox.world import build_world
from scenarios.catalog import SCENARIOS
from .human import CLIHuman, NoHuman
from .llm.factory import make_llm
from .policy import Policy
from .runner import run_task

ROOT = Path(__file__).resolve().parents[1]


def _run_id(label: str) -> Path:
    return ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"


def _print_result(r) -> None:
    print(f"\n=== {r.status.upper()} · verified={'yes' if r.verified else 'no'} · {r.steps} steps · "
          f"{r.input_tokens}+{r.output_tokens} tokens ===\n{r.summary}")
    for e in r.evidence:
        print(f"  - {e}")
    print(f"report: {r.run_dir / 'report.md'}")


def cmd_run(a) -> int:
    scenario = SCENARIOS[a.world]
    world = build_world(ROOT / "workspace", scenario.inbox, scenario.faults, port=a.port)
    print(f"BillTrack sandbox: {world.url}   world: {scenario.id}   model: ", end="")
    llm = make_llm(a.provider)
    print(llm.name)
    human = NoHuman() if a.non_interactive else CLIHuman()
    try:
        result = run_task(a.task, llm, human, Policy(a.approval), world.url, world.root, _run_id("task"),
                          headless=not a.headed, verify=not a.no_verify, max_steps=a.max_steps,
                          slow_mo_ms=400 if a.headed else 0)
    finally:
        world.close()
    _print_result(result)
    return 0 if result.status == "done" else 1


def cmd_eval(a) -> int:
    ids = a.ids or list(SCENARIOS)
    llm = make_llm(a.provider)
    rows = []
    for sid in ids:
        sc = SCENARIOS[sid]
        print(f"\n##### {sc.id}: {sc.title}\n#####   {sc.tests}")
        world = build_world(ROOT / "workspace" / sid, sc.inbox, sc.faults)
        human = sc.make_human()
        try:
            result = run_task(sc.goal, llm, human, Policy(sc.approval_mode), world.url, world.root,
                              _run_id(sid), headless=not a.headed, max_steps=a.max_steps)
            checks = sc.check(world, human, result)
        finally:
            world.close()
        if result.status == "llm_unavailable":
            rows.append({"scenario": sid, "passed": None, "status": result.status, "verified": False,
                         "steps": result.steps, "tokens": result.input_tokens + result.output_tokens,
                         "checks": [], "run_dir": str(result.run_dir)})
            print("--> ERROR (model unavailable: not a pass or a fail; rerun later)")
            continue
        status_ok = result.status in sc.expect_status
        checks = [(f"status in {sc.expect_status}", status_ok, result.status)] + checks
        passed = all(ok for _, ok, _ in checks)
        rows.append({"scenario": sid, "passed": passed, "status": result.status, "verified": result.verified,
                     "steps": result.steps, "tokens": result.input_tokens + result.output_tokens,
                     "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
                     "run_dir": str(result.run_dir)})
        print(f"--> {'PASS' if passed else 'FAIL'} ({result.steps} steps)")
        for n, ok, d in checks:
            print(f"    [{'x' if ok else ' '}] {n} {('- ' + d) if d and not ok else ''}")
    out = ROOT / "runs" / f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    for r in rows:
        label = {True: "PASS", False: "FAIL", None: "ERR "}[r["passed"]]
        print(f"{label}  {r['scenario']:<26} {r['status']:<12} {r['steps']:>3} steps  {r['tokens']:>7} tokens")
    ran = [r for r in rows if r["passed"] is not None]
    print(f"{sum(r['passed'] for r in ran)}/{len(ran)} scenarios passed"
          f"{f' ({len(rows) - len(ran)} not run: model unavailable)' if len(ran) < len(rows) else ''}. Details: {out}")
    return 0 if ran and len(ran) == len(rows) and all(r["passed"] for r in ran) else 1


def cmd_list(a) -> int:
    for s in SCENARIOS.values():
        print(f"{s.id:<26} {s.title}\n{'':<26} goal: {s.goal}\n")
    return 0


def cmd_serve(a) -> int:
    sc = SCENARIOS[a.world]
    world = build_world(ROOT / "workspace", sc.inbox, sc.faults, port=a.port)
    print(f"BillTrack running at {world.url} (Ctrl+C to stop). Inbox: {world.inbox}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        world.close()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="worker")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a task in the sandbox")
    r.add_argument("task")
    r.add_argument("--world", default="s1_latest_invoice", choices=list(SCENARIOS), help="which seeded sandbox")
    r.add_argument("--approval", default="risky", choices=["risky", "always", "never"])
    r.add_argument("--provider", choices=["gemini", "anthropic"])
    r.add_argument("--headed", action="store_true", help="show the browser")
    r.add_argument("--non-interactive", action="store_true", help="no human: questions unanswered, approvals denied")
    r.add_argument("--no-verify", action="store_true")
    r.add_argument("--max-steps", type=int, default=40)
    r.add_argument("--port", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("eval", help="run the eval scenarios with ground-truth checks")
    e.add_argument("ids", nargs="*", help=f"scenario ids (default: all): {', '.join(SCENARIOS)}")
    e.add_argument("--provider", choices=["gemini", "anthropic"])
    e.add_argument("--headed", action="store_true")
    e.add_argument("--max-steps", type=int, default=40)
    e.set_defaults(fn=cmd_eval)
    l = sub.add_parser("list", help="list scenarios")
    l.set_defaults(fn=cmd_list)
    s = sub.add_parser("serve", help="start only the sandbox app")
    s.add_argument("--world", default="s1_latest_invoice", choices=list(SCENARIOS))
    s.add_argument("--port", type=int, default=5055)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()
