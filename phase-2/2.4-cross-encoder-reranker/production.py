# requirements: langgraph==0.2.55 sentence-transformers==3.3.1 pydantic==2.11.3 structlog==24.4.0 numpy==2.1.3
from __future__ import annotations

import re
from typing import Literal, TypedDict

import numpy as np
import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from sentence_transformers import CrossEncoder


structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class RetrievalState(TypedDict):
    query: str
    candidates: list[dict[str, object]]
    selected_chunks: list[dict[str, object]]
    status: Literal["retrieved", "reranked", "selected"]


def retrieval_node(state: RetrievalState) -> RetrievalState:
    state["candidates"] = sorted(state["candidates"], key=lambda item: float(item["retrieval_score"]), reverse=True)[:8]
    state["status"] = "retrieved"
    return state


def reranker_node(state: RetrievalState) -> RetrievalState:
    pairs = [(state["query"], re.sub(r"ignore all previous instructions", "", str(item["text"]), flags=re.IGNORECASE)[:1500]) for item in state["candidates"]]
    scores = model.predict(pairs)
    rescored = []
    for item, score in zip(state["candidates"], np.asarray(scores, dtype=float), strict=True):
        rescored.append({**item, "rerank_score": float(score)})
    state["selected_chunks"] = sorted(rescored, key=lambda item: float(item["rerank_score"]), reverse=True)[:3]
    state["status"] = "reranked"
    return state


graph = StateGraph(RetrievalState)
graph.add_node("retrieve", retrieval_node)
graph.add_node("rerank", reranker_node)
graph.add_edge("retrieve", "rerank")
graph.set_entry_point("retrieve")
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("reranker.sqlite3"))
