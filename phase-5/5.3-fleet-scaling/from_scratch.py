# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

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


class LocalLimiterState(BaseModel):
    in_flight: int = Field(ge=0)
    max_concurrency: int = Field(gt=0)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


class LocalSemaphoreLimiter:
    def __init__(self, max_concurrency: int) -> None:
        self._state = LocalLimiterState(in_flight=0, max_concurrency=max_concurrency)
        self._semaphore = threading.Semaphore(max_concurrency)

    def __enter__(self) -> None:
        self._semaphore.acquire()
        self._state.in_flight += 1

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        self._state.in_flight -= 1
        self._semaphore.release()


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def call_provider(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("provider_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


def run_locally(prompt: str, limiter: LocalSemaphoreLimiter) -> str:
    with limiter:
        started_at = time.perf_counter()
        response = call_provider(prompt)
        log.info("local_call_complete", latency_ms=int((time.perf_counter() - started_at) * 1000))
        return response


if __name__ == "__main__":
    limiter = LocalSemaphoreLimiter(max_concurrency=4)
    run_locally("Explain why per-process semaphores fail across multiple pods.", limiter)
