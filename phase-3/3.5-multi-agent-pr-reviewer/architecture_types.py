from __future__ import annotations

from typing import Literal, TypedDict


class ReviewFinding(TypedDict):
    category: Literal["correctness", "security", "tests", "maintainability"]
    severity: Literal["low", "medium", "high"]
    finding: str


class ReviewState(TypedDict):
    diff_text: str
    correctness_findings: list[ReviewFinding]
    security_findings: list[ReviewFinding]
    test_findings: list[ReviewFinding]
    merged_review: str
    status: Literal["reviewed", "merged", "failed"]
