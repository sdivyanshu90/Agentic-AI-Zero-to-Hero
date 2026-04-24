from __future__ import annotations

from typing import Literal, TypedDict


class CandidateChunk(TypedDict):
    chunk_id: str
    text: str
    retrieval_score: float
    rerank_score: float


class RetrievalState(TypedDict):
    query: str
    candidates: list[CandidateChunk]
    selected_chunks: list[CandidateChunk]
    status: Literal["retrieved", "reranked", "selected"]
