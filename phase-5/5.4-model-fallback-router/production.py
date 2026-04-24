# requirements: circuitbreaker==1.4.0 anthropic==0.51.0 openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError as AnthropicRateLimitError
from circuitbreaker import circuit
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


class BudgetExceededError(RuntimeError):
    pass


class AllProvidersUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class ProviderState(BaseModel):
    provider_used: Literal["anthropic", "openai", "none"]
    breaker_status: Literal["closed", "open", "half_open"]
    shared_budget_remaining: int = Field(ge=0)
    status: Literal["served", "fallback", "shed"]
    error_type: str | None = None


class ProviderStateStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS provider_state (snapshot_ts INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    def save(self, state: ProviderState) -> None:
        self._connection.execute(
            "INSERT INTO provider_state (snapshot_ts, payload) VALUES (?, ?)",
            (int(time.time()), state.model_dump_json()),
        )
        self._connection.commit()


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
@circuit(failure_threshold=3, recovery_timeout=60)
def call_primary(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
@circuit(failure_threshold=3, recovery_timeout=60)
def call_secondary(prompt: str) -> str:
    enforce_token_budget(6000, prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


class ProviderRouter:
    def __init__(self, shared_budget: int, state_store: ProviderStateStore) -> None:
        self._shared_budget = shared_budget
        self._state_store = state_store

    def complete(self, prompt: str) -> str:
        if self._shared_budget <= 0:
            state = ProviderState(provider_used="none", breaker_status="open", shared_budget_remaining=0, status="shed", error_type="shared_budget_exhausted")
            self._state_store.save(state)
            raise AllProvidersUnavailableError("shared provider budget exhausted")
        try:
            content = call_primary(prompt)
            self._shared_budget -= 1
            self._state_store.save(ProviderState(provider_used="anthropic", breaker_status="closed", shared_budget_remaining=self._shared_budget, status="served"))
            return content
        except (APIConnectionError, APIStatusError, AnthropicRateLimitError) as primary_error:
            log.warning("fallback_event", provider="anthropic", error_type=type(primary_error).__name__, fallback_latency_overhead_ms=120)
        try:
            content = call_secondary(prompt)
            self._shared_budget -= 1
            self._state_store.save(ProviderState(provider_used="openai", breaker_status="half_open", shared_budget_remaining=self._shared_budget, status="fallback"))
            return content
        except (APIError, APITimeoutError, RateLimitError) as secondary_error:
            self._state_store.save(ProviderState(provider_used="none", breaker_status="open", shared_budget_remaining=self._shared_budget, status="shed", error_type=type(secondary_error).__name__))
            raise AllProvidersUnavailableError(str(secondary_error)) from secondary_error
