from __future__ import annotations

from typing import Literal, TypedDict


class FirewallRequest(TypedDict):
    request_id: str
    user_id: str
    source_ip: str
    message: str


class FirewallVerdict(TypedDict):
    decision: Literal["allow", "block", "review"]
    confidence: float
    matched_rules: list[str]
    reason: str


class FirewallState(TypedDict):
    request: FirewallRequest
    regex_verdict: FirewallVerdict
    llm_verdict: FirewallVerdict
    final_verdict: FirewallVerdict
    audit_key: str
