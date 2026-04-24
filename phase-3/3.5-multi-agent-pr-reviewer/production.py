# requirements: langgraph==0.2.55 langchain-openai==0.2.14 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
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


class ReviewState(TypedDict):
    diff_text: str
    correctness_findings: list[str]
    security_findings: list[str]
    test_findings: list[str]
    merged_review: str
    status: Literal["reviewed", "merged", "failed"]


def sanitize_diff(diff_text: str) -> str:
    diff_text = re.sub(r"ignore all previous instructions", "[REDACTED_INJECTION]", diff_text, flags=re.IGNORECASE)
    return diff_text[:12000]


def correctness_node(state: ReviewState) -> ReviewState:
    state["correctness_findings"] = [str(model.invoke(f"Find correctness issues in this diff: {sanitize_diff(state['diff_text'])}").content)]
    return state


def security_node(state: ReviewState) -> ReviewState:
    state["security_findings"] = [str(model.invoke(f"Find security issues in this diff: {sanitize_diff(state['diff_text'])}").content)]
    return state


def tests_node(state: ReviewState) -> ReviewState:
    state["test_findings"] = [str(model.invoke(f"Find missing or weak tests in this diff: {sanitize_diff(state['diff_text'])}").content)]
    return state


def merge_node(state: ReviewState) -> ReviewState:
    merged = state["correctness_findings"] + state["security_findings"] + state["test_findings"]
    state["merged_review"] = "\n".join(merged)
    state["status"] = "merged"
    return state


graph = StateGraph(ReviewState)
graph.add_node("correctness", correctness_node)
graph.add_node("security", security_node)
graph.add_node("tests", tests_node)
graph.add_node("merge", merge_node)
graph.set_entry_point("correctness")
graph.add_edge("correctness", "security")
graph.add_edge("security", "tests")
graph.add_edge("tests", "merge")
graph.add_edge("merge", END)
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("pr_reviewer.sqlite3"))
