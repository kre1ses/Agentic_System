"""
Anthropic Claude backend.
Returns native anthropic.types.Message objects — no conversion needed.

Retry policy:
  - Up to MAX_RETRIES attempts per model candidate.
  - Retries on RateLimitError and OverloadedError with exponential backoff.
  - If a fallback_model is provided and the primary exhausts retries, the
    fallback is tried with its own retry budget.
"""
import time
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, MAX_TOKENS

_MAX_RETRIES = 3          # attempts per model candidate
_BASE_WAIT   = 5          # seconds for first retry (doubles each time, cap 60s)


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
               max_tokens: int = MAX_TOKENS,
               fallback_model: str | None = None) -> Any:
        """
        Call the Anthropic API with automatic retry and optional model fallback.

        Retryable errors: RateLimitError, OverloadedError.
        On exhaustion of retries for the primary model, falls back to
        `fallback_model` (if provided and different from primary).
        """
        candidates = [model]
        if fallback_model and fallback_model != model:
            candidates.append(fallback_model)

        last_error: Exception | None = None
        for candidate in candidates:
            for attempt in range(_MAX_RETRIES):
                try:
                    return self._client.messages.create(
                        model=candidate,
                        max_tokens=max_tokens,
                        system=system,
                        tools=tools or [],
                        messages=messages,
                    )
                except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                    # APIStatusError covers overloaded (529) responses
                    is_retryable = isinstance(e, anthropic.RateLimitError) or (
                        hasattr(e, "status_code") and e.status_code in (429, 529)
                    )
                    if not is_retryable:
                        raise
                    wait = min(_BASE_WAIT * (2 ** attempt), 60)
                    print(f"    [llm/anthropic] {type(e).__name__} on {candidate}, "
                          f"retry {attempt + 1}/{_MAX_RETRIES} in {wait}s…")
                    time.sleep(wait)
                    last_error = e
                except Exception:
                    raise  # non-retryable — propagate immediately

            # Primary exhausted; if there's a fallback, announce it
            if len(candidates) > 1 and candidate == candidates[0]:
                print(f"    [llm/anthropic] Falling back to {candidates[1]}")

        raise RuntimeError(
            f"Anthropic: all candidates {candidates} exhausted. "
            f"Last error: {last_error}"
        )
