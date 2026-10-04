"""Writes report.md: outcome, verification, and an evidence trail for a finished run."""
from pathlib import Path


def write_report(agent, result) -> Path:
    L = [f"# Run report\n", f"**Goal:** {agent.goal}\n",
         f"**Outcome:** `{result.status}`  ·  independently verified: **{'yes' if result.verified else 'no'}**\n",
         f"**Summary:** {result.summary}\n"]
    if result.evidence:
        L.append("**Evidence cited by the worker:**\n" + "\n".join(f"- {e}" for e in result.evidence) + "\n")
    L.append(f"**Auditor verdict:** {result.verdict}\n")
    if agent.memory:
        L.append("## Notes the worker kept\n" + "\n".join(f"- `{k}`: {v}" for k, v in agent.memory.items()) + "\n")
    if agent.questions:
        L.append("## Questions asked\n" + "\n".join(f"- Q: {q}\n  A: {a or '(no answer)'}" for q, a in agent.questions) + "\n")
    if agent.approvals:
        L.append("## Approvals\n" + "\n".join(f"- {'granted' if ok else 'DENIED'}: {d}" for d, ok in agent.approvals) + "\n")
    L.append("## Actions\n" + "\n".join(
        f"{a['step']:>2}. {'ok ' if a['ok'] else 'ERR'} `{a['tool']}` {a.get('note', '')}"
        + (f"  **[state change: {a['intent']}]**" if a.get("mutating") else "") for a in agent.actions) + "\n")
    if agent.evidence_files:
        L.append("## Screenshots\n" + "\n".join(f"![{p.name}](evidence/{p.name})" for p in agent.evidence_files) + "\n")
    L.append(f"_{result.steps} steps, {result.input_tokens} input / {result.output_tokens} output tokens._\n")
    path = Path(result.run_dir) / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
