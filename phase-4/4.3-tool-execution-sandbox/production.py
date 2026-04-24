# requirements: langgraph==0.2.55 langchain-openai==0.2.14 docker==7.1.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import textwrap
import time
from dataclasses import dataclass
from typing import Literal, TypedDict

import docker
import requests
import structlog
from docker.errors import APIError as DockerAPIError, ContainerError, DockerException
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
docker_client = docker.from_env()
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, timeout=10.0)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class AgentState(TypedDict):
    task: str
    generated_code: str
    stdout: str
    stderr: str
    exit_code: int
    retry_count: int
    status: Literal["generate", "execute", "repair", "failed", "done"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_node(state: AgentState) -> AgentState:
    enforce_token_budget(5000, state["task"], state.get("stderr", ""))
    prompt = state["task"] if state["retry_count"] == 0 else f"Repair this code after error: {state['stderr']}"
    response = model.invoke(prompt)
    state["generated_code"] = str(response.content)
    state["status"] = "execute"
    return state


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def execute_node(state: AgentState) -> AgentState:
    start_time = time.perf_counter()
    try:
        output = docker_client.containers.run(
            image="python:3.11-alpine",
            command=["python", "-c", textwrap.dedent(state["generated_code"])],
            network_mode="none",
            read_only=True,
            mem_limit="64m",
            cpu_period=100000,
            cpu_quota=50000,
            remove=True,
            stderr=True,
            stdout=True,
            detach=False,
            volumes={},
        )
        state["stdout"] = output.decode("utf-8")
        state["stderr"] = ""
        state["exit_code"] = 0
        state["status"] = "done"
    except ContainerError as exc:
        state["stdout"] = ""
        state["stderr"] = str(exc)
        state["exit_code"] = 1
        state["retry_count"] += 1
        state["status"] = "repair"
    except requests.exceptions.ReadTimeout as exc:
        state["stdout"] = ""
        state["stderr"] = str(exc)
        state["exit_code"] = 124
        state["retry_count"] += 1
        state["status"] = "repair"
    except (DockerAPIError, DockerException) as exc:
        raise RuntimeError(str(exc)) from exc
    log.info("execute_node_complete", duration_ms=int((time.perf_counter() - start_time) * 1000), status=state["status"])
    return state


def route_after_execute(state: AgentState) -> str:
    if state["status"] == "done":
        return END
    if state["retry_count"] >= 3:
        return "fail"
    return "generate"


def fail_node(state: AgentState) -> AgentState:
    state["status"] = "failed"
    return state


graph = StateGraph(AgentState)
graph.add_node("generate", generate_node)
graph.add_node("execute", execute_node)
graph.add_node("fail", fail_node)
graph.set_entry_point("generate")
graph.add_edge("generate", "execute")
graph.add_conditional_edges("execute", route_after_execute, {END: END, "generate": "generate", "fail": "fail"})
compiled_graph = graph.compile(checkpointer=SqliteSaver.from_conn_string("sandbox_checkpoints.sqlite3"))
