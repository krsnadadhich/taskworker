import logging
import re
import time
import uuid

import httpx
from google import genai
from google.genai import errors, types

from .base import LLMError, Message, ModelMsg, ModelTurn, ResultsMsg, ToolCall, ToolSpec, UserMsg

log = logging.getLogger("worker.llm")
COOLDOWN_S = 45


class GeminiLLM:
    def __init__(self, api_key: str, models: list[str], thinking_level: str = "low",
                 timeout_s: int = 90, max_output_tokens: int = 4096):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1), timeout=timeout_s * 1000))
        self.models = models
        self.name = models[0]
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self._cooldown: dict[str, float] = {}

    def _contents(self, messages: list[Message]) -> list[types.Content]:
        out: list[types.Content] = []
        last_results = max((i for i, m in enumerate(messages) if isinstance(m, ResultsMsg)), default=-1)
        for i, m in enumerate(messages):
            if isinstance(m, UserMsg):
                out.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
            elif isinstance(m, ModelMsg):
                out.append(m.turn.raw)
            else:
                parts = [types.Part.from_function_response(name=r.name, response={"result": r.content})
                         for r in m.results]
                if m.note and i == last_results:
                    parts.append(types.Part(text=m.note))
                out.append(types.Content(role="user", parts=parts))
        return out

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.max_output_tokens,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description,
                                          parameters_json_schema=t.parameters) for t in tools])],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
        )
        contents = self._contents(messages)
        last_error: Exception | None = None
        for attempt in range(4):
            now = time.monotonic()
            ready = [m for m in self.models if self._cooldown.get(m, 0) <= now] or self.models
            for model in ready:
                try:
                    resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                except (errors.APIError, httpx.TimeoutException, httpx.TransportError) as e:
                    code = getattr(e, "code", "network")
                    log.warning("%s failed (%s)", model, code)
                    if isinstance(e, errors.APIError) and not (e.code in (408, 429) or e.code >= 500):
                        raise LLMError(f"{model}: {str(e)[:300]}") from e
                    self._cooldown[model] = time.monotonic() + COOLDOWN_S
                    last_error = e
                    continue
                turn = self._parse(resp)
                if turn:
                    return turn
                last_error = LLMError("empty model response")
            time.sleep(self._backoff(last_error, attempt))
        raise LLMError(f"all models failed: {str(last_error)[:300]}")

    @staticmethod
    def _backoff(err: Exception | None, attempt: int) -> float:
        """Rate limits: honour the server's retry hint when present, else wait long enough for a per-minute window."""
        if isinstance(err, errors.APIError) and err.code == 429:
            hint = re.search(r"retry(?: in |Delay'?: ?'?)([\d.]+)s", str(err))
            return min(max(float(hint.group(1)) if hint else 30.0 * (attempt + 1), 15.0), 70.0)
        return 2.0 * (attempt + 1)

    @staticmethod
    def _parse(resp) -> ModelTurn | None:
        if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
            return None
        content = resp.candidates[0].content
        calls, text = [], []
        for p in content.parts:
            if p.function_call:
                calls.append(ToolCall(p.function_call.id or uuid.uuid4().hex[:8], p.function_call.name,
                                      dict(p.function_call.args or {})))
            elif p.text and not p.thought:
                text.append(p.text)
        u = resp.usage_metadata
        return ModelTurn(" ".join(text).strip(), calls, raw=content,
                         input_tokens=(u.prompt_token_count or 0) if u else 0,
                         output_tokens=(u.candidates_token_count or 0) if u else 0)
