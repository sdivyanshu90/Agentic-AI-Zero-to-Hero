from __future__ import annotations

from typing import Literal, TypedDict


class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    revised_answer: str
    score: float
    retry_count: int
    status: Literal["drafted", "critiqued", "revised", "done", "failed"]
