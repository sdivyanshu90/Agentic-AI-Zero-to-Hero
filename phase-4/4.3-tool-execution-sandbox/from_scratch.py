# requirements: openai==1.77.0 docker==7.1.0 requests==2.32.3 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import textwrap
import time
from dataclasses import dataclass
from typing import Literal

import docker
import requests
import structlog
from docker.errors import APIError as DockerAPIError, ContainerError, DockerException, ImageNotFound
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
docker_client = docker.from_env()


class BudgetExceededError(RuntimeError):
    pass


class ExecutionFailedError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int = Field(ge=0)


CODEGEN_PROMPT: str = """
Write Python 3.11 code that solves the task.
Do not read local files, do not use sockets, and write the final answer to stdout.
Return code only.
""".strip()


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_code(task: str) -> str:
    enforce_token_budget(5000, CODEGEN_PROMPT, task)
    try:
        completion = llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": CODEGEN_PROMPT},
                {"role": "user", "content": task},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("codegen_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or "raise RuntimeError('empty code output')"


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def execute_code_in_sandbox(code: str) -> ExecutionResult | ToolError:
    start_time = time.perf_counter()
    try:
        container_output = docker_client.containers.run(
            image="python:3.11-alpine",
            command=["python", "-c", textwrap.dedent(code)],
            network_mode="none",
            read_only=True,
            mem_limit="64m",
            cpu_period=100000,
            cpu_quota=50000,
            remove=True,
            detach=False,
            stderr=True,
            stdout=True,
        )
    except ContainerError as exc:
        return ToolError(tool_name="execute_code_in_sandbox", error_type="ContainerError", message=str(exc))
    except requests.exceptions.ReadTimeout as exc:
        return ToolError(tool_name="execute_code_in_sandbox", error_type="ReadTimeout", message=str(exc))
    except (DockerAPIError, DockerException, ImageNotFound) as exc:
        raise ExecutionFailedError(str(exc)) from exc
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    return ExecutionResult(stdout=container_output.decode("utf-8"), stderr="", exit_code=0, duration_ms=duration_ms)


def solve_task(task: str, max_repairs: int = 3) -> ExecutionResult:
    retry_attempt_count = 0
    current_task = task
    while retry_attempt_count < max_repairs:
        generated_code = generate_code(current_task)
        execution_result = execute_code_in_sandbox(generated_code)
        if isinstance(execution_result, ToolError):
            retry_attempt_count += 1
            current_task = f"Original task: {task}\nExecution failed with: {execution_result.message}\nRepair the code."
            continue
        return execution_result
    raise ExecutionFailedError("sandbox execution failed after 3 repair attempts")


if __name__ == "__main__":
    result = solve_task("Write Python that prints the first ten prime numbers, one line each.")
    log.info("sandbox_complete", result=result.model_dump())
