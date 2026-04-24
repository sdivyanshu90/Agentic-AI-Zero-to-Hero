# requirements: langgraph==0.2.55 langchain-openai==0.2.14 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
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
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", os.getcwd())).resolve()


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class AgentState(TypedDict):
    messages: list[dict[str, str]]
    pending_write: bool
    status: Literal["running", "awaiting_write_approval", "done", "failed"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def agent_node(state: AgentState) -> AgentState:
    serialized_messages = json.dumps(state["messages"], ensure_ascii=True)
    enforce_token_budget(6000, serialized_messages)
    response = model.invoke(state["messages"])
    content = str(response.content)
    state["messages"].append({"role": "assistant", "content": content})
    state["pending_write"] = "write_file" in content
    state["status"] = "awaiting_write_approval" if state["pending_write"] else "done"
    return state


def route_after_agent(state: AgentState) -> str:
    return "human_review" if state["pending_write"] else END


def human_review_node(state: AgentState) -> AgentState:
    log.warning("write_requires_human_approval")
    state["status"] = "awaiting_write_approval"
    return state


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("human_review", human_review_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {END: END, "human_review": "human_review"})
graph.add_edge("human_review", END)
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("cli_assistant.sqlite3"), interrupt_before=["human_review"])
