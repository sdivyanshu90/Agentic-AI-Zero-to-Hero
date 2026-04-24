from __future__ import annotations

from typing import Literal, TypedDict


class GraphQuery(TypedDict):
    question: str
    schema_text: str


class GraphResult(TypedDict):
    cypher: str
    rows: list[dict[str, object]]


class GraphState(TypedDict):
    question: str
    cypher: str
    retry_count: int
    status: Literal["generated", "executed", "failed"]
