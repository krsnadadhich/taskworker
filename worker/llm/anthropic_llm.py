import uuid

import anthropic

from .base import LLMError, Message, ModelMsg, ModelTurn, ResultsMsg, ToolCall, ToolSpec, UserMsg


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_output_tokens: int = 4096):
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=90)
        self.name = model
        self.max_output_tokens = max_output_tokens

    def _messages(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        last_results = max((i for i, m in enumerate(messages) if isinstance(m, ResultsMsg)), default=-1)
        for i, m in enumerate(messages):
            if isinstance(m, UserMsg):
                out.append({"role": "user", "content": m.text})
            elif isinstance(m, ModelMsg):
                out.append({"role": "assistant", "content": m.turn.raw})
            else:
                blocks = [{"type": "tool_result", "tool_use_id": r.call_id, "content": r.content,
                           "is_error": r.is_error} for r in m.results]
                if m.note and i == last_results:
                    blocks.append({"type": "text", "text": m.note})
                out.append({"role": "user", "content": blocks})
        return out

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        try:
            resp = self.client.messages.create(
                model=self.name, system=system, max_tokens=self.max_output_tokens,
                messages=self._messages(messages),
                tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
                tool_choice={"type": "any"})
        except anthropic.APIError as e:
            raise LLMError(str(e)[:300]) from e
        raw = [b.model_dump(exclude_none=True) for b in resp.content]
        calls = [ToolCall(b.id or uuid.uuid4().hex[:8], b.name, dict(b.input)) for b in resp.content
                 if b.type == "tool_use"]
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        if not calls:
            raise LLMError("model returned no tool call")
        return ModelTurn(text, calls, raw=raw, input_tokens=resp.usage.input_tokens,
                         output_tokens=resp.usage.output_tokens)
