# requirements: fastapi==0.115.12 redis==5.2.1 openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import redis
import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from redis.exceptions import ConnectionError as RedisConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("REDIS_URL"):
    raise EnvironmentError("REDIS_URL not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)

RATE_LIMIT = 3000
WINDOW_SECONDS = 60


class BudgetExceededError(RuntimeError):
    pass


class OverCapacityError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class RateLimitState(BaseModel):
    request_id: str
    window_count: int = Field(ge=0)
    delayed_ms: int = Field(ge=0)
    status: Literal["allowed", "delayed", "shed"]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def redis_client() -> redis.Redis:
    return redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True, max_connections=100)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def acquire_slot(request_id: str) -> RateLimitState:
    client = redis_client()
    key = f"rate_limit:{int(time.time() // WINDOW_SECONDS)}"
    try:
        pipeline = client.pipeline()
        pipeline.incr(key, 1)
        pipeline.expire(key, WINDOW_SECONDS)
        window_count, _ = pipeline.execute()
    except RedisConnectionError as exc:
        log.warning("redis_retry", error_type=type(exc).__name__, message=str(exc))
        raise

    if int(window_count) > RATE_LIMIT:
        return RateLimitState(request_id=request_id, window_count=int(window_count), delayed_ms=0, status="shed")
    if int(window_count) > int(0.85 * RATE_LIMIT):
        delayed_ms = int(random.uniform(50, 250))
        time.sleep(delayed_ms / 1000)
        return RateLimitState(request_id=request_id, window_count=int(window_count), delayed_ms=delayed_ms, status="delayed")
    return RateLimitState(request_id=request_id, window_count=int(window_count), delayed_ms=0, status="allowed")


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def call_provider(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


def handle_request(request_id: str, prompt: str) -> str:
    state = acquire_slot(request_id)
    if state.status == "shed":
        raise OverCapacityError("shared rate limit exhausted")
    return call_provider(prompt)
