from __future__ import annotations

from typing import Literal, TypedDict


class IntentClassification(TypedDict):
    intent: Literal["ORDER_STATUS", "REFUND", "TECH_SUPPORT", "UNKNOWN"]


class RouteResult(TypedDict):
    intent: str
    handler_output: str


class RouterState(TypedDict):
    user_input: str
    normalized_intent: str
    route_result: RouteResult
    status: Literal["classified", "routed", "failed"]
