from __future__ import annotations

from typing import Literal, TypedDict


class MemoryFact(TypedDict):
    key: str
    value: str
    confidence: float


class ConversationState(TypedDict):
    user_id: str
    user_input: str
    retrieved_facts: list[MemoryFact]
    extracted_facts: list[MemoryFact]
    status: Literal["loaded", "responded", "persisted"]
