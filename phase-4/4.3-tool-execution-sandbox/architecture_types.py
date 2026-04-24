from __future__ import annotations

from typing import Literal, TypedDict


class ExecutionRequest(TypedDict):
    task: str
    generated_code: str
    timeout_seconds: int


class ExecutionResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


class ExecutionState(TypedDict):
    request: ExecutionRequest
    result: ExecutionResult
    status: Literal["generated", "executed", "repaired", "failed"]
    retry_count: int
