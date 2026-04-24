from __future__ import annotations

from typing import Literal, TypedDict


class AgentState(TypedDict):
    question: str
    retrieved_context: str
    answer: str
    status: Literal["planned", "retrieved", "answered", "failed"]
    retry_count: int


class ToolResult(TypedDict):
    content: str
    source: str
