# requirements: langgraph==0.2.55 langchain-openai==0.2.14 tiktoken==0.7.0 pydantic==2.11.3 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
from typing import Literal, TypedDict

import structlog
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, timeout=10.0)
encoder = tiktoken.encoding_for_model("gpt-4.1-mini")


class WindowState(TypedDict):
    messages: list[dict[str, object]]
    summary: str
    max_prompt_tokens: int
    status: Literal["assembled", "summarized", "trimmed"]


def selector_node(state: WindowState) -> WindowState:
    for message in state["messages"]:
        message["token_count"] = len(encoder.encode(str(message["content"])))
    state["status"] = "assembled"
    return state


def summary_node(state: WindowState) -> WindowState:
    low_priority = [message for message in state["messages"] if int(message["priority"]) < 8]
    response = model.invoke(f"Summarize these messages and preserve critical constraints only: {low_priority}")
    state["summary"] = str(response.content)
    state["status"] = "summarized"
    return state


def trim_node(state: WindowState) -> WindowState:
    state["messages"] = [message for message in state["messages"] if int(message["priority"]) >= 8] + [
        {"role": "system", "content": state["summary"], "priority": 9, "token_count": len(encoder.encode(state["summary"]))}
    ] + [message for message in state["messages"][-4:] if int(message["priority"]) < 8]
    state["status"] = "trimmed"
    return state


graph = StateGraph(WindowState)
graph.add_node("selector", selector_node)
graph.add_node("summary", summary_node)
graph.add_node("trim", trim_node)
graph.add_edge("selector", "summary")
graph.add_edge("summary", "trim")
graph.set_entry_point("selector")
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("context_window.sqlite3"))
