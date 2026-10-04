from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Intent:
    """A state-changing action a tool is about to take, described before it happens."""
    kind: str
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class ToolOutput:
    text: str
    ok: bool = True
    ephemeral: bool = False
    evidence: list[Path] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], ToolOutput]
    preflight: Callable[[dict], Intent | None] | None = None
    read_only: bool = False  # available to the independent verifier


def schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


def missing_args(tool: Tool, args: dict) -> list[str]:
    return [r for r in tool.parameters.get("required", []) if r not in args]
