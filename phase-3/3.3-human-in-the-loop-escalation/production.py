# requirements: langgraph==0.2.55 fastapi==0.115.12 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
app = FastAPI()


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class ApprovalState(TypedDict):
    thread_id: str
    proposed_action: str
    action_hash: str
    final_action: str
    decision: Literal["approve", "edit", "reject", "pending"]
    risk_level: Literal["low", "medium", "high"]
    status: Literal["drafted", "waiting_human", "executed", "rejected"]


class ApprovalPayload(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    reviewer_id: str = Field(min_length=1)
    edited_action: str = ""
    action_hash: str = Field(min_length=1)


def compute_action_hash(thread_id: str, proposed_action: str) -> str:
    return hashlib.sha256(f"{thread_id}:{proposed_action}".encode("utf-8")).hexdigest()


def approval_node(state: ApprovalState) -> ApprovalState:
    state["status"] = "waiting_human"
    return state


def execution_node(state: ApprovalState) -> ApprovalState:
    if state["decision"] == "reject":
        state["status"] = "rejected"
        return state
    state["final_action"] = state["final_action"] or state["proposed_action"]
    state["status"] = "executed"
    return state


def route_after_approval(state: ApprovalState) -> str:
    return END if state["status"] in {"executed", "rejected"} else "execute"


graph = StateGraph(ApprovalState)
graph.add_node("approval", approval_node)
graph.add_node("execute", execution_node)
graph.set_entry_point("approval")
graph.add_conditional_edges("approval", route_after_approval, {END: END, "execute": "execute"})
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("human_approval.sqlite3"), interrupt_before=["approval"])


@app.post("/threads/{thread_id}/approve")
def approve(thread_id: str, payload: ApprovalPayload) -> dict[str, str]:
    saved_state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    if saved_state is None:
        raise HTTPException(status_code=404, detail="thread not found")
    state = saved_state.values
    expected_hash = compute_action_hash(thread_id, state["proposed_action"])
    if payload.action_hash != expected_hash:
        raise HTTPException(status_code=409, detail="approval does not match current action")
    state["decision"] = payload.decision
    state["final_action"] = payload.edited_action
    compiled_graph.update_state({"configurable": {"thread_id": thread_id}}, state)
    return {"thread_id": thread_id, "status": payload.decision}
