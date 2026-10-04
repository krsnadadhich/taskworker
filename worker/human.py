"""The human in the loop: clarification questions and approvals."""
from typing import Callable, Protocol

from .tools.base import Intent


class Human(Protocol):
    def ask(self, question: str, options: list[str] | None) -> str | None: ...
    def approve(self, intent: Intent, reason: str) -> bool: ...


class NoHuman:
    """Non-interactive run: nobody to ask, so risky actions are refused and questions go unanswered."""

    def ask(self, question, options):
        return None

    def approve(self, intent, reason):
        return False


class CLIHuman:
    def ask(self, question, options):
        print(f"\n  ? {question}")
        if options:
            print("    options: " + " | ".join(options))
        try:
            return input("    > ").strip() or None
        except EOFError:
            return None

    def approve(self, intent, reason):
        print(f"\n  ! Approval needed ({reason})\n    {intent.description}")
        for k, v in intent.details.get("fields", {}).items():
            print(f"      {k}: {v}")
        try:
            return input("    approve? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False


class ScriptedHuman:
    """For evals/tests: answers questions via a callable and approves via a flag."""

    def __init__(self, answer: Callable[[str, list[str] | None], str | None] | None = None, approve: bool = True):
        self.answer = answer
        self.approves = approve
        self.questions: list[str] = []
        self.approval_requests: list[str] = []

    def ask(self, question, options):
        self.questions.append(question)
        return self.answer(question, options) if self.answer else None

    def approve(self, intent, reason):
        self.approval_requests.append(intent.description)
        return self.approves
