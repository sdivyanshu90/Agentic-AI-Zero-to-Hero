# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Literal

import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class StreamEvent(BaseModel):
    event: Literal["tool_start", "tool_end", "agent_message", "error", "done"]
    data: str
    sequence_id: int = Field(ge=0)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
async def stream_completion(prompt: str) -> AsyncIterator[StreamEvent]:
    enforce_token_budget(6000, prompt)
    started_at = time.perf_counter()
    try:
        stream = await client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("stream_retry", error_type=type(exc).__name__, message=str(exc))
        raise

    yield StreamEvent(event="tool_start", data="llm_stream", sequence_id=0)
    sequence_id = 1
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield StreamEvent(event="agent_message", data=delta, sequence_id=sequence_id)
            sequence_id += 1
    yield StreamEvent(event="tool_end", data=f"ttft_ms={int((time.perf_counter() - started_at) * 1000)}", sequence_id=sequence_id)
    yield StreamEvent(event="done", data="stream_complete", sequence_id=sequence_id + 1)


async def main() -> None:
    async for event in stream_completion("Explain why retries need jitter in distributed systems."):
        log.info("stream_event", payload=event.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
