# requirements: langgraph==0.2.55 langchain-openai==0.2.14 neo4j==5.27.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("NEO4J_URI"):
    raise EnvironmentError("NEO4J_URI not set")
if not os.getenv("NEO4J_USERNAME"):
    raise EnvironmentError("NEO4J_USERNAME not set")
if not os.getenv("NEO4J_PASSWORD"):
    raise EnvironmentError("NEO4J_PASSWORD not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, timeout=10.0)
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class GraphState(TypedDict):
    question: str
    schema_text: str
    allowed_relationships: list[str]
    cypher: str
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


def generator_node(state: GraphState) -> GraphState:
    prompt = f"Schema:\n{state['schema_text']}\nQuestion:\n{state['question']}\nPrevious error:\n{state['error_text']}"
    enforce_token_budget(6000, prompt)
    response = model.invoke(prompt)
    state["cypher"] = str(response.content).strip().strip("`")
    state["status"] = "generated"
    return state


def executor_node(state: GraphState) -> GraphState:
    for relationship_name in re.findall(r":([A-Z_]+)\]", state["cypher"]):
        if relationship_name not in set(state["allowed_relationships"]):
            state["error_text"] = f"invalid relationship type {relationship_name}"
            state["retry_count"] += 1
            state["status"] = "failed"
            return state
    try:
        with driver.session() as session:
            list(session.run(state["cypher"]))
        state["status"] = "executed"
    except CypherSyntaxError as exc:
        state["error_text"] = str(exc)
        state["retry_count"] += 1
        state["status"] = "failed"
    return state


def route_after_execution(state: GraphState) -> str:
    if state["status"] == "executed" or state["retry_count"] >= 3:
        return END
    return "generator"


graph = StateGraph(GraphState)
graph.add_node("generator", generator_node)
graph.add_node("executor", executor_node)
graph.set_entry_point("generator")
graph.add_edge("generator", "executor")
graph.add_conditional_edges("executor", route_after_execution, {END: END, "generator": "generator"})
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("graphrag.sqlite3"))
