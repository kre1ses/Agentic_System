"""
LLM client factory.

Reads LLM_PROVIDER from environment / config and returns the right backend.

Supported providers:
  anthropic    — Claude models (requires ANTHROPIC_API_KEY)
  openrouter   — Open-source models via OpenRouter (requires OPENROUTER_API_KEY)
  vsegpt       — VseGPT proxy (requires VSEGPT_API_KEY)
  huggingface  — HF Serverless Inference API (requires HF_TOKEN)
  none         — No LLM; agents use rule-based fallbacks
"""
import os
from typing import Any


def get_llm_client() -> Any | None:
    """
    Return an LLM backend instance based on LLM_PROVIDER env var,
    or None if no provider is configured (triggers rule-based fallback).
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect from available keys if provider not explicitly set
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.environ.get("VSEGPT_API_KEY"):
            provider = "vsegpt"
        elif os.environ.get("HF_TOKEN"):
            provider = "huggingface"
        else:
            return None

    if provider == "anthropic":
        from llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if provider == "openrouter":
        from llm.openai_compat_backend import OpenAICompatBackend
        return OpenAICompatBackend(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://github.com/multi-agent-system",
                "X-Title": "Multi-Agent ML System",
            },
        )

    if provider == "vsegpt":
        from llm.openai_compat_backend import OpenAICompatBackend
        return OpenAICompatBackend(
            api_key=os.environ["VSEGPT_API_KEY"],
            base_url="https://api.vsegpt.ru/v1",
        )

    if provider == "huggingface":
        from llm.hf_backend import HuggingFaceBackend
        return HuggingFaceBackend(
            token=os.environ["HF_TOKEN"],
            provider=os.environ.get("HF_PROVIDER", "auto"),
        )

    if provider in ("none", "no-llm", ""):
        return None

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. "
                     f"Choose: anthropic | openrouter | vsegpt | huggingface | none")
