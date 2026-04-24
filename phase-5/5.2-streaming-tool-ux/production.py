# requirements: fastapi==0.115.12 sse-starlette==2.1.3 langgraph==0.2.55 langchain-openai==0.2.14 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Literal, TypedDict

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from sse_starlette.sse import EventSourceResponse


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, timeout=10.0)
app = FastAPI()


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class StreamState(TypedDict):
    prompt: str
    final_answer: str
    status: Literal["streaming", "failed", "done"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def agent_node(state: StreamState) -> StreamState:
    enforce_token_budget(6000, state["prompt"])
    response = model.invoke(state["prompt"])
    state["final_answer"] = str(response.content)
    state["status"] = "done"
    return state


def route_after_agent(state: StreamState) -> str:
    return END if state["status"] == "done" else "error"


def error_node(state: StreamState) -> StreamState:
    state["status"] = "failed"
    state["final_answer"] = "stream failed"
    return state


graph = StateGraph(StreamState)
graph.add_node("agent", agent_node)
graph.add_node("error", error_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {END: END, "error": "error"})
graph.add_edge("error", END)
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("stream_checkpoints.sqlite3"))


@app.get("/stream")
async def stream(prompt: str) -> EventSourceResponse:
    async def event_generator() -> AsyncIterator[dict[str, str]]:
        event_index = 0
        async for event in compiled_graph.astream_events({"prompt": prompt, "final_answer": "", "status": "streaming"}, version="v1"):
            encoded_data = json.dumps(event, ensure_ascii=True).replace("\n", "\\n")
            event_type = "agent_message" if event.get("event") == "on_chat_model_stream" else "tool_end"
            yield {"event": event_type, "data": encoded_data, "id": str(event_index)}
            event_index += 1
        yield {"event": "done", "data": "stream_complete", "id": str(event_index)}

    return EventSourceResponse(event_generator(), headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})
