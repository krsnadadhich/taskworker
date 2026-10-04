"""Wires tools, policy, human and LLM into an Agent for one task."""
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from .agent import Agent, Config, RunResult
from .human import Human
from .llm.base import LLM
from .policy import Policy
from .tools.browser import BrowserSession
from .tools.files import Workspace
from .trace import Trace

ENVIRONMENT = """You work inside a sandboxed company environment (no real systems or credentials).
- A web application, "BillTrack", the company's internal accounts-payable system, at {url}. Explore it with the
  browser tools to find your way around. The browser can only reach {host}.
- A workspace folder with incoming documents (invoices, emails). Use list_files / read_file; paths are relative to the
  workspace root, e.g. "inbox"."""


def run_task(goal: str, llm: LLM, human: Human, policy: Policy, app_url: str, workspace: Path, run_dir: Path,
             *, headless: bool = True, verify: bool = True, echo: bool = True, max_steps: int = 40,
             slow_mo_ms: int = 0) -> RunResult:
    trace = Trace(run_dir, echo=echo)
    browser = BrowserSession(run_dir / "evidence", {urlparse(app_url).netloc}, headless=headless, slow_mo_ms=slow_mo_ms)
    try:
        tools = browser.tools() + Workspace(workspace).tools()
        agent = Agent(llm, tools, human, policy, trace, ENVIRONMENT.format(url=app_url, host=urlparse(app_url).netloc),
                      date.today().isoformat(), Config(max_steps=max_steps, verify=verify),
                      final_screenshot=lambda: browser.screenshot("final"))
        return agent.run(goal)
    finally:
        browser.close()
        trace.close()
