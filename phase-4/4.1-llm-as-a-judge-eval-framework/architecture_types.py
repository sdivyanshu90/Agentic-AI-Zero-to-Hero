from __future__ import annotations

from typing import Literal, TypedDict


class EvalCase(TypedDict):
    input: str
    expected: str
    tags: list[str]


class JudgeScore(TypedDict):
    precision: float
    recall: float
    faithfulness: float
    tone: float
    weighted_mean: float
    reason: str


class EvalState(TypedDict):
    run_id: str
    case_index: int
    candidate_output: str
    score: JudgeScore
    threshold: float
    status: Literal["pending", "running", "failed", "passed"]
