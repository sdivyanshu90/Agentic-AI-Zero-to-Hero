from __future__ import annotations

from typing import Literal, TypedDict


class TraceSpan(TypedDict):
    name: str
    latency_ms: int
    had_error: bool
    prompt_tokens: int
    completion_tokens: int


class TraceState(TypedDict):
    thread_id: str
    agent_type: str
    spans: list[TraceSpan]
    export_status: Literal["pending", "exported", "failed"]
    total_tokens: int
    total_latency_ms: int
