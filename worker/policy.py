"""Approval policy: decides whether a state-changing action needs a human yes before it runs."""
import re
from dataclasses import dataclass

from .tools.base import Intent

RISKY_WORDS = re.compile(r"\b(pay|payment|approve|send|transfer|delete|remove|cancel|void|wire|refund|submit payment)\b", re.I)


@dataclass
class Decision:
    needs_approval: bool
    reason: str


class Policy:
    """mode: 'risky' (default) asks only for risky actions; 'always' asks for every state change; 'never' asks for none."""

    def __init__(self, mode: str = "risky", amount_threshold: float = 10_000.0):
        assert mode in ("risky", "always", "never")
        self.mode = mode
        self.amount_threshold = amount_threshold

    def evaluate(self, intent: Intent) -> Decision:
        if self.mode == "always":
            return Decision(True, "policy: approve every state change")
        if self.mode == "never":
            return Decision(False, "policy: auto-approve")
        label = intent.details.get("label", "") + " " + intent.details.get("action", "")
        if RISKY_WORDS.search(label):
            return Decision(True, "irreversible or money-moving action")
        for key, value in intent.details.get("fields", {}).items():
            if "amount" in key.lower():
                try:
                    if float(str(value).replace(",", "")) >= self.amount_threshold:
                        return Decision(True, f"amount at or above {self.amount_threshold:,.0f}")
                except ValueError:
                    pass
        return Decision(False, "routine reversible change")
