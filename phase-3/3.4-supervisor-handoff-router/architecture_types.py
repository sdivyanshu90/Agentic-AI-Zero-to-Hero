from __future__ import annotations

from typing import Literal, TypedDict


class HandoffState(TypedDict):
    user_request: str
    route: Literal["billing", "security", "docs", "none"]
    specialist_response: str
    retry_count: int
    status: Literal["routed", "handled", "failed"]
