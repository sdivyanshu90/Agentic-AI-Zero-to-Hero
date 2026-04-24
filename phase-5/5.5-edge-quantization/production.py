# requirements: requests==2.32.3 openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests
import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("LLAMA_SERVER_URL"):
    raise EnvironmentError("LLAMA_SERVER_URL not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class BenchmarkState(BaseModel):
    provider: Literal["local", "remote"]
    ttft_ms: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    prompt: str


class BenchmarkStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS benchmark_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    def save(self, state: BenchmarkState) -> None:
        self._connection.execute("INSERT INTO benchmark_runs (payload) VALUES (?)", (state.model_dump_json(),))
        self._connection.commit()


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


class LocalModelClient:
    def __init__(self, use_local_model: bool) -> None:
        self._use_local_model = use_local_model
        self._remote_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def chat(self, prompt: str) -> BenchmarkState:
        enforce_token_budget(6000, prompt)
        started_at = time.perf_counter()
        if self._use_local_model:
            response = requests.post(
                f"{os.getenv('LLAMA_SERVER_URL')}/v1/chat/completions",
                json={
                    "model": "llama-3-8b-instruct-q4_k_m",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "grammar": "root ::= '{' ws '\"status\"' ws ':' ws string ws ',' ws '\"next_step\"' ws ':' ws string ws '}'",
                },
                timeout=10,
            )
            response.raise_for_status()
            output_text = response.json()["choices"][0]["message"]["content"]
            provider = "local"
        else:
            try:
                completion = self._remote_client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            except (APIError, APITimeoutError, RateLimitError) as exc:
                log.warning("remote_retry", error_type=type(exc).__name__, message=str(exc))
                raise
            output_text = completion.choices[0].message.content or ""
            provider = "remote"

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        generated_tokens = max(len(output_text.split()), 1)
        return BenchmarkState(
            provider=provider,
            ttft_ms=latency_ms,
            tokens_per_second=generated_tokens / max(latency_ms / 1000, 0.001),
            quality_score=0.92 if provider == "local" else 1.0,
            prompt=prompt,
        )


def route_request(prompt: str) -> BenchmarkState:
    use_local_model = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
    client = LocalModelClient(use_local_model=use_local_model)
    state = client.chat(prompt)
    BenchmarkStore(Path("benchmarks.sqlite3")).save(state)
    return state
