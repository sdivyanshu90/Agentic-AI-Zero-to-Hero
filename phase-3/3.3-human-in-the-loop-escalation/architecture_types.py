from __future__ import annotations

from typing import Literal, TypedDict


class ApprovalRequest(TypedDict):
    thread_id: str
    proposed_action: str
    risk_level: Literal["low", "medium", "high"]


class ApprovalDecision(TypedDict):
    decision: Literal["approve", "edit", "reject"]
    edited_action: str
    reviewer_id: str


class ApprovalState(TypedDict):
    thread_id: str
    proposed_action: str
    final_action: str
    decision: Literal["approve", "edit", "reject", "pending"]
    status: Literal["drafted", "waiting_human", "executed", "rejected"]
