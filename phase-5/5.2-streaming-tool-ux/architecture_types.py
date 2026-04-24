from __future__ import annotations

from typing import Literal, TypedDict


class StreamEvent(TypedDict):
    event: Literal["tool_start", "tool_end", "agent_message", "error", "done"]
    data: str
    sequence_id: int


class StreamState(TypedDict):
    thread_id: str
    events: list[StreamEvent]
    final_answer: str
    status: Literal["streaming", "failed", "done"]
