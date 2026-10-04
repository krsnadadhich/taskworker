import os
from pathlib import Path

from dotenv import load_dotenv

from .base import LLM

ROOT = Path(__file__).resolve().parents[2]


def make_llm(provider: str | None = None) -> LLM:
    load_dotenv(ROOT / ".env")
    provider = (provider or os.getenv("WORKER_PROVIDER", "gemini")).lower()
    if provider == "gemini":
        from .gemini import GeminiLLM
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set (put it in .env, see .env.example)")
        models = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite").split(",")]
        return GeminiLLM(key, models, thinking_level=os.getenv("GEMINI_THINKING", "low"))
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set (put it in .env, see .env.example)")
        return AnthropicLLM(key, os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    raise SystemExit(f"unknown provider {provider!r} (use gemini or anthropic)")
