# requirements: requests==2.32.3 openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

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
remote_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class BenchmarkResult(BaseModel):
    provider: str
    ttft_ms: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0.0)
    output_text: str


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def run_command(command: list[str], timeout_seconds: int) -> str | ToolError:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
        return completed.stdout
    except subprocess.CalledProcessError as exc:
        return ToolError(tool_name=command[0], error_type="CalledProcessError", message=exc.stderr)
    except subprocess.TimeoutExpired as exc:
        return ToolError(tool_name=command[0], error_type="TimeoutExpired", message=str(exc))


def quantize_model(model_path: Path, gguf_path: Path) -> str | ToolError:
    return run_command([
        "llama-quantize",
        str(model_path),
        str(gguf_path),
        "Q4_K_M",
    ], timeout_seconds=600)


def start_server(gguf_path: Path) -> str | ToolError:
    return run_command([
        "llama-server",
        "-m",
        str(gguf_path),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ], timeout_seconds=5)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def call_local(prompt: str) -> BenchmarkResult:
    enforce_token_budget(6000, prompt)
    started_at = time.perf_counter()
    response = requests.post(
        f"{os.getenv('LLAMA_SERVER_URL')}/v1/chat/completions",
        json={
            "model": "llama-3-8b-instruct-q4_k_m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
        },
        timeout=10,
    )
    response.raise_for_status()
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    payload = response.json()
    output_text = payload["choices"][0]["message"]["content"]
    generated_tokens = max(len(output_text.split()), 1)
    return BenchmarkResult(provider="local", ttft_ms=latency_ms, tokens_per_second=generated_tokens / max(latency_ms / 1000, 0.001), output_text=output_text)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def call_remote(prompt: str) -> BenchmarkResult:
    enforce_token_budget(6000, prompt)
    started_at = time.perf_counter()
    try:
        response = remote_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("remote_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    output_text = response.choices[0].message.content or ""
    generated_tokens = max(len(output_text.split()), 1)
    return BenchmarkResult(provider="remote", ttft_ms=latency_ms, tokens_per_second=generated_tokens / max(latency_ms / 1000, 0.001), output_text=output_text)


if __name__ == "__main__":
    quantize_result = quantize_model(Path("models/llama-3-8b-instruct-f16.gguf"), Path("models/llama-3-8b-instruct-q4_k_m.gguf"))
    if isinstance(quantize_result, ToolError):
        raise RuntimeError(quantize_result.message)
    local_benchmark = call_local("Return a JSON object with keys status and next_step for a failed deploy.")
    remote_benchmark = call_remote("Return a JSON object with keys status and next_step for a failed deploy.")
    log.info("benchmark_complete", local=local_benchmark.model_dump(), remote=remote_benchmark.model_dump())
