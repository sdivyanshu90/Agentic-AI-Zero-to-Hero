# requirements: langgraph==0.2.55 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1 openai==1.77.0
from __future__ import annotations

import base64
import binascii
import codecs
import os
import re
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class FilterState(TypedDict):
    response: str
    confidential_corpus: list[str]
    filtered_response: str
    redaction_events: list[dict[str, int | str]]
    blocked: bool
    verdict: Literal["safe", "redacted", "human_review", "blocked"]


PATTERNS: dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "api_key": r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "internal_ip": r"\b(?:10|127|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)(?:\.\d{1,3}){2}\b",
    "base64_blob": r"[A-Za-z0-9+/]{40,}={0,2}",
}


def decode_variants(response: str) -> list[str]:
    variants = [response]
    try:
        variants.append(codecs.decode(response, "rot_13"))
    except UnicodeDecodeError:
        pass
    for candidate in re.findall(PATTERNS["base64_blob"], response):
        try:
            decoded_bytes = base64.b64decode(candidate, validate=True)
            variants.append(decoded_bytes.decode("utf-8", errors="ignore"))
        except (binascii.Error, ValueError):
            continue
    return variants


def filter_node(state: FilterState) -> FilterState:
    filtered_response = state["response"]
    redaction_events: list[dict[str, int | str]] = []
    for pattern_name, pattern in PATTERNS.items():
        for match in re.finditer(pattern, filtered_response):
            redaction_events.append(
                {"pattern_name": pattern_name, "start_offset": match.start(), "end_offset": match.end()}
            )
            log.warning("redaction_event", pattern_name=pattern_name, start_offset=match.start(), end_offset=match.end())
        filtered_response = re.sub(pattern, f"[REDACTED_{pattern_name.upper()}]", filtered_response)

    blocked = False
    for variant in decode_variants(state["response"]):
        for corpus_entry in state["confidential_corpus"]:
            left_tokens = set(variant[:512].split())
            right_tokens = set(corpus_entry[:512].split())
            similarity = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
            if similarity > 0.6:
                blocked = True
                break
        if blocked:
            break

    state["filtered_response"] = filtered_response
    state["redaction_events"] = redaction_events
    state["blocked"] = blocked
    state["verdict"] = "human_review" if blocked else ("redacted" if redaction_events else "safe")
    return state


def route_filter(state: FilterState) -> str:
    return "human_review" if state["blocked"] else END


def human_review_node(state: FilterState) -> FilterState:
    state["verdict"] = "blocked"
    state["filtered_response"] = "Response withheld pending security review."
    return state


graph = StateGraph(FilterState)
graph.add_node("filter", filter_node)
graph.add_node("human_review", human_review_node)
graph.set_entry_point("filter")
graph.add_conditional_edges("filter", route_filter, {END: END, "human_review": "human_review"})
graph.add_edge("human_review", END)
compiled_graph = graph.compile(
    checkpointer=SqliteSaver.from_conn_string("filter_checkpoints.sqlite3"),
    interrupt_before=["human_review"],
)
