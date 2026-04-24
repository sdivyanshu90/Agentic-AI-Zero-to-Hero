from __future__ import annotations

from typing import Literal, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    score: float


class SupportAnswer(TypedDict):
    answer: str
    cited_chunk_ids: list[str]


class RAGState(TypedDict):
    query: str
    retrieved_chunks: list[RetrievedChunk]
    answer: SupportAnswer
    status: Literal["retrieved", "validated", "failed"]
