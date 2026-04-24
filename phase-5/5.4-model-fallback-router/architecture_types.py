from __future__ import annotations

from typing import Literal, TypedDict


class ProviderRequest(TypedDict):
    prompt: str
    max_tokens: int


class ProviderState(TypedDict):
    provider_used: Literal["anthropic", "openai", "none"]
    breaker_status: Literal["closed", "open", "half_open"]
    shared_budget_remaining: int
    status: Literal["served", "fallback", "shed"]
    error_type: str | None
