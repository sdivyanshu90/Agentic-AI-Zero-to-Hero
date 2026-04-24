from __future__ import annotations

from typing import Literal, TypedDict


class TaskRequest(TypedDict):
    task_id: str
    prompt: str
    user_id: str


class TaskState(TypedDict):
    task_id: str
    status: Literal["Pending", "Running", "Completed", "Failed"]
    result: str | None
    error: str | None
    retry_count: int


class DeadLetterRecord(TypedDict):
    task_id: str
    prompt: str
    final_error: str
