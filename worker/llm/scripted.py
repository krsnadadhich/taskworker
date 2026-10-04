"""Deterministic stand-in LLM for tests: replays a fixed list of turns (or callables)."""
import itertools

from .base import LLMError, Message, ModelTurn, ToolCall, ToolSpec

_ids = itertools.count(1)


def turn(name: str, text: str = "", **args) -> ModelTurn:
    return ModelTurn(text, [ToolCall(f"c{next(_ids)}", name, args)])


class ScriptedLLM:
    name = "scripted"

    def __init__(self, script: list):
        self.script = list(script)
        self.seen: list[list[Message]] = []

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        self.seen.append(list(messages))
        if not self.script:
            raise LLMError("script exhausted")
        nxt = self.script.pop(0)
        return nxt(messages) if callable(nxt) else nxt
