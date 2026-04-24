# requirements: fastapi==0.115.12 celery==5.4.0 redis==5.2.1 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1 openai==1.77.0 uvicorn==0.34.2
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Literal

import redis
import structlog
from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("REDIS_URL"):
    raise EnvironmentError("REDIS_URL not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
celery_app = Celery("agent_queue", broker=os.getenv("REDIS_URL"), backend=os.getenv("REDIS_URL"))
app = FastAPI()


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class TaskRecord(BaseModel):
    task_id: str
    status: Literal["Pending", "Running", "Completed", "Failed"]
    result: str | None = None
    error: str | None = None
    retry_count: int = Field(ge=0, default=0)


class TaskCreateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def redis_client() -> redis.Redis:
    return redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)


def save_task(record: TaskRecord) -> None:
    redis_client().set(f"task:{record.task_id}", record.model_dump_json(), ex=86400)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def run_agent(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


@celery_app.task(bind=True, autoretry_for=(APIError, APITimeoutError, RateLimitError), retry_backoff=True, retry_jitter=True, max_retries=3)
def execute_agent_task(self: object, task_id: str, prompt: str) -> str:
    record = TaskRecord(task_id=task_id, status="Running", retry_count=int(self.request.retries))
    save_task(record)
    try:
        result = run_agent(prompt)
        record.status = "Completed"
        record.result = result
        save_task(record)
        return result
    except MaxRetriesExceededError as exc:
        record.status = "Failed"
        record.error = str(exc)
        save_task(record)
        redis_client().lpush("dead_letter", json.dumps({"task_id": task_id, "prompt": prompt, "final_error": str(exc)}))
        raise


@app.post("/tasks")
def create_task(payload: TaskCreateRequest) -> dict[str, str]:
    task_id = str(uuid.uuid4())
    save_task(TaskRecord(task_id=task_id, status="Pending"))
    execute_agent_task.delay(task_id=task_id, prompt=payload.prompt)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}/status")
def get_task_status(task_id: str) -> TaskRecord:
    raw_record = redis_client().get(f"task:{task_id}")
    if raw_record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskRecord.model_validate_json(raw_record)


@app.get("/tasks/dlq")
def get_dead_letter_queue() -> list[dict[str, str]]:
    return [json.loads(item) for item in redis_client().lrange("dead_letter", 0, 100)]
