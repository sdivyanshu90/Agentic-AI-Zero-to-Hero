# requirements: anthropic==0.51.0 openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
from dataclasses import dataclass

import structlog
from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError as AnthropicRateLimitError
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("ANTHROPIC_API_KEY"):
    raise EnvironmentError("ANTHROPIC_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class BudgetExceededError(RuntimeError):
    pass


class AllProvidersUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class CompletionResult(BaseModel):
    provider: str
    content: str
    fallback_used: bool = False
    latency_overhead_ms: int = Field(ge=0, default=0)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def call_primary(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except (APIConnectionError, APIStatusError, AnthropicRateLimitError) as exc:
        log.warning("primary_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return response.content[0].text


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def call_secondary(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("secondary_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return response.choices[0].message.content or ""


def complete_with_fallback(prompt: str) -> CompletionResult:
    try:
        return CompletionResult(provider="anthropic", content=call_primary(prompt))
    except (APIConnectionError, APIStatusError, AnthropicRateLimitError) as primary_error:
        log.warning("fallback_triggered", provider="anthropic", error_type=type(primary_error).__name__)
    try:
        return CompletionResult(provider="openai", content=call_secondary(prompt), fallback_used=True, latency_overhead_ms=120)
    except (APIError, APITimeoutError, RateLimitError) as secondary_error:
        raise AllProvidersUnavailableError(str(secondary_error)) from secondary_error


if __name__ == "__main__":
    result = complete_with_fallback("Summarize the deployment rollback policy.")
    log.info("completion_complete", result=result.model_dump())
