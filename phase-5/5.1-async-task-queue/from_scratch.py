# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Literal

import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class TaskState(BaseModel):
    task_id: str
    prompt: str
    status: Literal["Pending", "Running", "Completed", "Failed"] = "Pending"
    result: str | None = None
    error: str | None = None
    retry_count: int = Field(ge=0, default=0)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def run_agent(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("agent_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


class ThreadQueueRunner:
    def __init__(self) -> None:
        self._queue: queue.Queue[TaskState] = queue.Queue()
        self._tasks: dict[str, TaskState] = {}
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def submit(self, prompt: str) -> str:
        task_state = TaskState(task_id=str(uuid.uuid4()), prompt=prompt)
        self._tasks[task_state.task_id] = task_state
        self._queue.put(task_state)
        return task_state.task_id

    def status(self, task_id: str) -> TaskState:
        return self._tasks[task_id]

    def _work(self) -> None:
        while True:
            task_state = self._queue.get()
            task_state.status = "Running"
            try:
                task_state.result = run_agent(task_state.prompt)
                task_state.status = "Completed"
            except (APIError, APITimeoutError, RateLimitError) as exc:
                task_state.error = str(exc)
                task_state.retry_count += 1
                task_state.status = "Failed"
            finally:
                self._queue.task_done()


if __name__ == "__main__":
    runner = ThreadQueueRunner()
    task_id = runner.submit("Summarize the incident timeline in four bullet points.")
    log.info("task_submitted", task_id=task_id, status=runner.status(task_id).status)
