from __future__ import annotations

from typing import Literal, TypedDict


class RateLimitRequest(TypedDict):
    request_id: str
    pod_name: str
    prompt: str


class RateLimitState(TypedDict):
    request_id: str
    window_count: int
    delayed_ms: int
    status: Literal["allowed", "delayed", "shed"]


class ProviderResult(TypedDict):
    response_text: str
    provider_status: int
