"""Provider-neutral message and tool types. Providers translate these to their native API."""
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    ephemeral: bool = False  # bulky observation (page snapshot) that may be elided once stale


@dataclass
class ModelTurn:
    text: str
    calls: list[ToolCall]
    raw: Any = None  # provider-native object, replayed verbatim (e.g. Gemini thought signatures)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class UserMsg:
    text: str


@dataclass
class ModelMsg:
    turn: ModelTurn


@dataclass
class ResultsMsg:
    results: list[ToolResult]
    note: str = ""  # per-turn state block; only the latest one is sent


Message = UserMsg | ModelMsg | ResultsMsg


class LLMError(Exception):
    """Provider unavailable or returned nothing usable after retries."""


class LLM(Protocol):
    name: str

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn: ...
