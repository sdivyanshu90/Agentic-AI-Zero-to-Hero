# requirements: langgraph==0.2.55 langchain-openai==0.2.14 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("LANGCHAIN_API_KEY"):
    raise EnvironmentError("LANGCHAIN_API_KEY not set")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agentic-ai-zero-to-hero"

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


class TraceState(TypedDict):
    thread_id: str
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    had_error: bool
    needs_alert: bool
    status: Literal["running", "observed", "alerted"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def agent_node(state: TraceState) -> TraceState:
    enforce_token_budget(6000, state["prompt"])
    response = model.invoke(state["prompt"])
    response_text = str(response.content)
    state["response"] = response_text
    state["prompt_tokens"] = estimate_prompt_tokens(state["prompt"])
    state["completion_tokens"] = estimate_prompt_tokens(response_text)
    state["latency_ms"] = 250
    state["had_error"] = False
    state["needs_alert"] = state["latency_ms"] > 5000 or state["prompt_tokens"] > 20000
    state["status"] = "observed"
    return state


def route_trace(state: TraceState) -> str:
    return "alert" if state["needs_alert"] or state["had_error"] else END


def alert_node(state: TraceState) -> TraceState:
    log.warning(
        "trace_alert",
        thread_id=state["thread_id"],
        latency_ms=state["latency_ms"],
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
    )
    state["status"] = "alerted"
    return state


graph = StateGraph(TraceState)
graph.add_node("agent", agent_node)
graph.add_node("alert", alert_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_trace, {END: END, "alert": "alert"})
graph.add_edge("alert", END)
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("trace_checkpoints.sqlite3"))
