"""
OpenAI-compatible backend.

Works with:
  - OpenRouter  (OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
  - VseGPT      (VSEGPT_API_KEY,     base_url="https://api.vsegpt.ru/v1")
  - Any service exposing /v1/chat/completions

Converts Anthropic-style tool definitions and messages to OpenAI format,
then wraps the response in FakeResponse so base_agent.py is unchanged.
"""
import json
import time
import uuid
from typing import Any

from config import MAX_TOKENS
from llm.types import FakeResponse, FakeTextBlock, FakeToolUseBlock

# Ordered fallback list: if the primary model is 429'd, try the next one
_FALLBACK_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-4b:free",
    "google/gemma-3-27b-it:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
]


class OpenAICompatBackend:
    """OpenAI-compatible backend (OpenRouter / VseGPT / etc.)."""

    PROVIDER = "openai_compat"

    def __init__(self, api_key: str, base_url: str,
                 extra_headers: dict | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_headers = extra_headers or {}

    # ------------------------------------------------------------------
    # Public interface matching base_agent.py expectations
    # ------------------------------------------------------------------

    def create(self, model: str, system: str,
               messages: list[dict], tools: list[dict],
               max_tokens: int = MAX_TOKENS) -> FakeResponse:
        oai_messages = self._to_oai_messages(system, messages)
        oai_tools    = self._to_oai_tools(tools) if tools else None

        kwargs: dict[str, Any] = dict(
            messages=oai_messages,
            max_tokens=max_tokens,
        )
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers

        # Try primary model, then fallbacks on 429
        candidates = [model] + [m for m in _FALLBACK_MODELS if m != model]
        last_error: Exception | None = None
        for attempt, candidate in enumerate(candidates):
            try:
                resp = self._client.chat.completions.create(
                    model=candidate, **kwargs
                )
                if candidate != model:
                    print(f"    [llm] Fell back to {candidate}")
                return self._to_fake_response(resp)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    wait = min(4 * (attempt + 1), 30)
                    print(f"    [llm] 429 on {candidate}, waiting {wait}s...")
                    time.sleep(wait)
                    last_error = e
                    continue
                raise  # re-raise non-429 errors immediately
        raise RuntimeError(
            f"All models exhausted after rate limits. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Format conversion: Anthropic → OpenAI
    # ------------------------------------------------------------------

    @staticmethod
    def _to_oai_tools(anthropic_tools: list[dict]) -> list[dict]:
        """Anthropic tool schema → OpenAI function schema."""
        oai = []
        for t in anthropic_tools:
            oai.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return oai

    @staticmethod
    def _to_oai_messages(system: str, messages: list[dict]) -> list[dict]:
        """
        Convert Anthropic message list (with possible tool_result content blocks)
        to OpenAI message list.
        """
        oai: list[dict] = []
        if system:
            oai.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                oai.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                # Check if this is a tool-result message (role=user with tool_result blocks)
                if role == "user" and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                ):
                    for block in content:
                        oai.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        })
                    continue

                # Assistant message with text + tool_use blocks
                text_parts = []
                tool_calls = []
                for block in content:
                    btype = getattr(block, "type", None) or (
                        block.get("type") if isinstance(block, dict) else None
                    )
                    if btype == "text":
                        text = getattr(block, "text", None) or block.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif btype == "tool_use":
                        bid   = getattr(block, "id",   None) or block.get("id",   str(uuid.uuid4())[:8])
                        bname = getattr(block, "name", None) or block.get("name", "")
                        binput = getattr(block, "input", None) or block.get("input", {})
                        tool_calls.append({
                            "id": bid,
                            "type": "function",
                            "function": {
                                "name": bname,
                                "arguments": json.dumps(binput),
                            },
                        })

                asst_msg: dict[str, Any] = {"role": "assistant"}
                asst_msg["content"] = " ".join(text_parts) if text_parts else None
                if tool_calls:
                    asst_msg["tool_calls"] = tool_calls
                oai.append(asst_msg)
                continue

            # Fallback
            oai.append({"role": role, "content": str(content)})

        return oai

    # ------------------------------------------------------------------
    # Format conversion: OpenAI → Fake Anthropic response
    # ------------------------------------------------------------------

    @staticmethod
    def _to_fake_response(resp) -> FakeResponse:
        choice = resp.choices[0]
        finish = choice.finish_reason           # "stop" | "tool_calls" | "length"
        msg    = choice.message

        content: list[Any] = []

        if msg.content:
            content.append(FakeTextBlock(text=msg.content))

        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            content.append(FakeToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=args,
            ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return FakeResponse(stop_reason=stop_reason, content=content)
