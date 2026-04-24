from __future__ import annotations

from typing import Literal, TypedDict


class ManagedMessage(TypedDict):
    role: str
    content: str
    priority: int
    token_count: int


class WindowState(TypedDict):
    summary: str
    selected_messages: list[ManagedMessage]
    dropped_count: int
    status: Literal["assembled", "summarized", "trimmed"]
