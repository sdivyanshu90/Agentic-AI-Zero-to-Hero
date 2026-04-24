from __future__ import annotations

from typing import Literal, TypedDict


class FilterRequest(TypedDict):
    response_text: str
    confidential_corpus: list[str]


class RedactionEvent(TypedDict):
    pattern_name: str
    start_offset: int
    end_offset: int


class FilterState(TypedDict):
    request: FilterRequest
    redactions: list[RedactionEvent]
    blocked: bool
    verdict: Literal["safe", "redacted", "human_review", "blocked"]
    filtered_response: str
