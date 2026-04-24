# requirements: langgraph==0.2.55 langchain-openai==0.2.14 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import sqlite3
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


class QueryState(TypedDict):
    question: str
    schema_text: str
    query: str
    error_text: str
    retry_count: int
    status: Literal["generated", "executed", "failed"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def agent_node(state: QueryState) -> QueryState:
    prompt = f"Schema:\n{state['schema_text']}\nQuestion:\n{state['question']}\nPrevious error:\n{state['error_text']}"
    enforce_token_budget(6000, prompt)
    response = model.invoke(prompt)
    state["query"] = str(response.content).strip().strip("`")
    state["status"] = "generated"
    return state


def executor_node(state: QueryState) -> QueryState:
    if not state["query"].strip().lower().startswith("select"):
        state["error_text"] = "only SELECT queries are allowed"
        state["retry_count"] += 1
        state["status"] = "failed"
        return state
    connection = sqlite3.connect("app.db")
    try:
        connection.execute(state["query"])
        state["status"] = "executed"
    except sqlite3.OperationalError as exc:
        state["error_text"] = str(exc)
        state["retry_count"] += 1
        state["status"] = "failed"
    return state


def route_after_execution(state: QueryState) -> str:
    if state["status"] == "executed":
        return END
    if state["retry_count"] >= 3:
        return END
    return "agent"


graph = StateGraph(QueryState)
graph.add_node("agent", agent_node)
graph.add_node("executor", executor_node)
graph.set_entry_point("agent")
graph.add_edge("agent", "executor")
graph.add_conditional_edges("executor", route_after_execution, {END: END, "agent": "agent"})
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("sql_generator.sqlite3"))
