"""
Base agent class.

Supports multiple LLM backends (Anthropic, OpenRouter, VseGPT, HuggingFace)
via the llm.factory module. All backends return FakeResponse-compatible objects
so the ReAct tool loop is backend-agnostic.
"""
import json
import time
from typing import Any

from config import MAX_TOKENS, MAX_TOOL_RETRIES
from llm.factory import get_llm_client
from memory.experiment_store import ExperimentStore
from rag.knowledge_base import KnowledgeBase
from safety.guardrails import Guardrails


class BaseAgent:
    """
    Abstract base agent.

    Subclasses must set:
        self.name         – display name
        self.role         – system-prompt role description
        self.tools        – list of Claude-style tool schemas (dicts)
        self._dispatchers – list of objects with .dispatch(name, input)
    """

    def __init__(
        self,
        model: str,
        kb: KnowledgeBase | None = None,
        store: ExperimentStore | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.kb = kb or KnowledgeBase()
        self.store = store or ExperimentStore()
        self.verbose = verbose
        self.name = "BaseAgent"
        self.role = "You are a helpful assistant."
        self.tools: list[dict] = []
        self._dispatchers: list[Any] = []
        self._client = get_llm_client()     # None → rule-based fallback

    # ------------------------------------------------------------------
    # Core run method (ReAct loop — provider-agnostic)
    # ------------------------------------------------------------------

    def run(self, user_message: str, rag_query: str | None = None,
            extra_context: str = "") -> str:
        """Execute the agent; returns final text response."""
        self._log(f"Starting with: {user_message[:120]}...")

        system = self._build_system(rag_query or user_message, extra_context)

        if self._client is None:
            return self._fallback(user_message)

        messages: list[dict] = [{"role": "user", "content": user_message}]

        for attempt in range(MAX_TOOL_RETRIES + 1):
            try:
                response = self._client.create(
                    model=self.model,
                    system=system,
                    messages=messages,
                    tools=self.tools or [],
                    max_tokens=MAX_TOKENS,
                )
            except Exception as e:
                self._log(f"API error: {e}", level="error")
                time.sleep(min(2 ** attempt, 30))
                continue

            # ── ReAct tool-use loop ────────────────────────────────────
            while response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    btype = getattr(block, "type", None)
                    if btype != "tool_use":
                        continue
                    result = self._call_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user",      "content": tool_results})

                response = self._client.create(
                    model=self.model,
                    system=system,
                    messages=messages,
                    tools=self.tools or [],
                    max_tokens=MAX_TOKENS,
                )

            final_text = self._extract_text(response)

            safe, _ = Guardrails.validate_agent_response(final_text)
            if not safe:
                self._log("Output blocked by guardrails.", level="warn")
                final_text = "[Response blocked by output guardrails]"

            self.store.log_message(f"[{self.name}] {final_text[:300]}", agent=self.name)
            self._log("Done.")
            return final_text

        return f"[{self.name}] Failed after {MAX_TOOL_RETRIES} retries."

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _call_tool(self, name: str, tool_input: dict) -> Any:
        safe, reason = Guardrails.validate_tool_input(name, tool_input)
        if not safe:
            self._log(f"Tool input rejected: {reason}", level="warn")
            return {"error": reason}

        for dispatcher in self._dispatchers:
            result = dispatcher.dispatch(name, tool_input)
            if not (isinstance(result, dict) and
                    result.get("error") == f"Unknown tool: {name}"):
                self._log(f"Tool '{name}' -> {str(result)[:80]}")
                return result
        return {"error": f"No dispatcher for tool: {name}"}

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system(self, rag_query: str, extra_context: str = "") -> str:
        parts = [self.role]
        if extra_context:
            parts.append(f"\nAdditional context:\n{extra_context}")
        # Role-aware retrieval: use agent name as role key
        agent_role = self.name.lower()
        rag_ctx = self.kb.retrieve_as_context(
            rag_query, agent_role=agent_role
        )
        if rag_ctx:
            parts.append(f"\n{rag_ctx}")
        past = self.store.get_context_for_rag()
        if past:
            parts.append(f"\n{past}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
        return "\n".join(parts).strip()

    def _fallback(self, message: str) -> str:
        from config import ACTIVE_LLM_PROVIDER
        return (f"[{self.name} -- no LLM ({ACTIVE_LLM_PROVIDER})] "
                f"Received: {message[:200]}")

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {"info": "->", "warn": "[!]", "error": "[x]"}.get(level, "->")
            print(f"  [{self.name}] {prefix} {msg}")
