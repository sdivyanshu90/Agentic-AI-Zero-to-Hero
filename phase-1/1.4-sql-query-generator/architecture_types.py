from __future__ import annotations

from typing import Literal, TypedDict


class QueryRequest(TypedDict):
    question: str
    schema_text: str


class SQLResult(TypedDict):
    query: str
    rows: list[dict[str, object]]


class QueryState(TypedDict):
    question: str
    query: str
    retry_count: int
    status: Literal["generated", "executed", "failed"]
