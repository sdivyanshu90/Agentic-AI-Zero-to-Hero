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


class HandoffState(TypedDict):
    user_request: str
    route: Literal["billing", "security", "docs", "none"]
    specialist_response: str
    retry_count: int
    status: Literal["routed", "handled", "failed"]


def supervisor_node(state: HandoffState) -> HandoffState:
    request = state["user_request"].casefold()
    if any(word in request for word in ["refund", "invoice", "charge"]):
        state["route"] = "billing"
    elif any(word in request for word in ["token", "secret", "permission"]):
        state["route"] = "security"
    elif any(word in request for word in ["docs", "guide", "api"]):
        state["route"] = "docs"
    else:
        state["route"] = "none"
    state["status"] = "routed"
    return state


def billing_node(state: HandoffState) -> HandoffState:
    state["specialist_response"] = str(model.invoke(f"Billing specialist: {state['user_request']}").content)
    state["status"] = "handled"
    return state


def security_node(state: HandoffState) -> HandoffState:
    state["specialist_response"] = str(model.invoke(f"Security specialist: {state['user_request']}").content)
    state["status"] = "handled"
    return state


def docs_node(state: HandoffState) -> HandoffState:
    state["specialist_response"] = str(model.invoke(f"Docs specialist: {state['user_request']}").content)
    state["status"] = "handled"
    return state


def route_specialist(state: HandoffState) -> str:
    if state["route"] == "billing":
        return "billing"
    if state["route"] == "security":
        return "security"
    if state["route"] == "docs":
        return "docs"
    state["status"] = "failed"
    return END


graph = StateGraph(HandoffState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("billing", billing_node)
graph.add_node("security", security_node)
graph.add_node("docs", docs_node)
graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_specialist, {"billing": "billing", "security": "security", "docs": "docs", END: END})
graph.add_edge("billing", END)
graph.add_edge("security", END)
graph.add_edge("docs", END)
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("supervisor_router.sqlite3"))
