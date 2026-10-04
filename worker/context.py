"""Context management: keep the transcript small by eliding stale bulky observations."""
from dataclasses import replace

from .llm.base import Message, ResultsMsg


def _elide(content: str) -> str:
    head = "\n".join(content.splitlines()[:3])
    return f"{head}\n[older page snapshot elided to save context; call browser_read if you need it again]"


def compact(messages: list[Message], keep: int = 1) -> list[Message]:
    """Replace ephemeral tool results with a stub, except in the newest `keep` result messages that have any."""
    with_ephemeral = [i for i, m in enumerate(messages)
                      if isinstance(m, ResultsMsg) and any(r.ephemeral for r in m.results)]
    stale = set(with_ephemeral[:-keep] if keep else with_ephemeral)
    out: list[Message] = []
    for i, m in enumerate(messages):
        if i in stale:
            m = ResultsMsg([replace(r, content=_elide(r.content)) if r.ephemeral else r for r in m.results], m.note)
        out.append(m)
    return out
