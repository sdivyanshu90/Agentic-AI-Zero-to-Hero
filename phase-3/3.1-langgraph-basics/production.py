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


class AgentState(TypedDict):
    question: str
    retrieved_context: str
    answer: str
    status: Literal["planned", "retrieved", "answered", "failed"]
    retry_count: int


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def retrieve_node(state: AgentState) -> AgentState:
    state["retrieved_context"] = f"Context for: {state['question']}"
    state["status"] = "retrieved"
    return state


def answer_node(state: AgentState) -> AgentState:
    enforce_token_budget(6000, state["question"], state["retrieved_context"])
    response = model.invoke(f"Question: {state['question']}\nContext: {state['retrieved_context']}")
    state["answer"] = str(response.content)
    state["status"] = "answered"
    return state


def route_after_answer(state: AgentState) -> str:
    if state["answer"]:
        return END
    state["retry_count"] += 1
    return END if state["retry_count"] >= 2 else "retrieve"


graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_conditional_edges("answer", route_after_answer, {END: END, "retrieve": "retrieve"})
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("langgraph_basics.sqlite3"))
