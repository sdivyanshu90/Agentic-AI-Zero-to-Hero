from __future__ import annotations

from typing import Literal, TypedDict


class ToolCall(TypedDict):
    tool_name: Literal["bash_exec", "read_file", "write_file"]
    arguments_json: str


class ToolResult(TypedDict):
    tool_name: str
    success: bool
    output: str


class AgentState(TypedDict):
    messages: list[dict[str, str]]
    tool_results: list[ToolResult]
    iteration_count: int
    status: Literal["running", "awaiting_write_approval", "done", "failed"]
