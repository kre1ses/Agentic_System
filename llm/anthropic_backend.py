"""
Anthropic Claude backend.
Returns native anthropic.types.Message objects — no conversion needed.
"""
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, MAX_TOKENS


class AnthropicBackend:
    """
    Thin wrapper; `create()` signature matches what base_agent.py expects.
    Provider: Anthropic API  (paid, best quality)
    """
    PROVIDER = "anthropic"

    def __init__(self, api_key: str | None = None):
        key = api_key or ANTHROPIC_API_KEY
        self._client = anthropic.Anthropic(api_key=key)

    # ── Public interface (mirrors self._client.messages) ────────────────
    @property
    def messages(self):
        return self._client.messages

    def create(self, model: str, system: str,
               messages: list[dict], tools: list[dict],
               max_tokens: int = MAX_TOKENS) -> Any:
        return self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools or [],
            messages=messages,
        )
