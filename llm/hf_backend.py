"""
HuggingFace Serverless Inference API backend.

Uses `huggingface_hub.InferenceClient` which exposes an OpenAI-compatible
`chat_completion` method. Tool calling is supported for models that have it
(e.g. Qwen2.5-72B-Instruct, Llama-3.1-8B-Instruct).

Env var: HF_TOKEN  (from huggingface.co → Settings → Access Tokens)
Docs:    https://huggingface.co/learn/cookbook/enterprise_hub_serverless_inference_api
"""
import json
from typing import Any

from config import MAX_TOKENS
from llm.types import FakeResponse, FakeTextBlock, FakeToolUseBlock


class HuggingFaceBackend:
    """HuggingFace Serverless Inference API (free for public models)."""

    PROVIDER = "huggingface"

    def __init__(self, token: str, provider: str = "auto"):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("pip install huggingface_hub")
        # provider="auto" lets HF pick the fastest serverless provider
        self._client = InferenceClient(token=token, provider=provider)

    def create(self, model: str, system: str,
               messages: list[dict], tools: list[dict],
               max_tokens: int = MAX_TOKENS) -> FakeResponse:
        """Call HF Inference API and return a FakeResponse."""
        from llm.openai_compat_backend import OpenAICompatBackend

        oai_msgs  = OpenAICompatBackend._to_oai_messages(system, messages)
        oai_tools = OpenAICompatBackend._to_oai_tools(tools) if tools else None

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=oai_msgs,
            max_tokens=max_tokens,
        )
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat_completion(**kwargs)
        return OpenAICompatBackend._to_fake_response(resp)
