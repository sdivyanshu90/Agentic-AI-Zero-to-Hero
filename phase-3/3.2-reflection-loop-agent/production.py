# requirements: langgraph==0.2.55 langchain-openai==0.2.14 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, timeout=10.0)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class ReflectionState(TypedDict):
    task: str
    current_answer: str
    critique: str
    score: float
    retry_count: int
    status: Literal["drafted", "critiqued", "revised", "done", "failed"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def draft_node(state: ReflectionState) -> ReflectionState:
    enforce_token_budget(6000, state["task"])
    response = model.invoke(state["task"])
    state["current_answer"] = str(response.content)
    state["status"] = "drafted"
    return state


def critique_node(state: ReflectionState) -> ReflectionState:
    response = model.invoke(f"Critique this answer and give a numeric score from 0.0 to 1.0:\nTask: {state['task']}\nAnswer: {state['current_answer']}")
    critique_text = str(response.content)
    state["critique"] = critique_text
    state["score"] = 0.90 if "good" in critique_text.casefold() else 0.60
    state["status"] = "critiqued"
    return state


def revise_node(state: ReflectionState) -> ReflectionState:
    response = model.invoke(f"Revise this answer using the critique only:\nAnswer: {state['current_answer']}\nCritique: {state['critique']}")
    state["current_answer"] = str(response.content)
    state["retry_count"] += 1
    state["status"] = "revised"
    return state


def route_after_critique(state: ReflectionState) -> str:
    if state["score"] >= 0.85:
        state["status"] = "done"
        return END
    if state["retry_count"] >= 2:
        state["status"] = "failed"
        return END
    return "revise"


graph = StateGraph(ReflectionState)
graph.add_node("draft", draft_node)
graph.add_node("critique", critique_node)
graph.add_node("revise", revise_node)
graph.set_entry_point("draft")
graph.add_edge("draft", "critique")
graph.add_conditional_edges("critique", route_after_critique, {END: END, "revise": "revise"})
graph.add_edge("revise", "critique")
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("reflection.sqlite3"))
