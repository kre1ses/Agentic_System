"""
Shared data types that all LLM backends return.

We mimic enough of the Anthropic response structure so that
base_agent.py works unchanged regardless of which backend is used.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class FakeResponse:
    """
    Mimics anthropic.types.Message well enough for base_agent.py.
    stop_reason: "end_turn" | "tool_use"
    content: list of FakeTextBlock | FakeToolUseBlock
    """
    stop_reason: str
    content: list[Any]
