# requirements: opentelemetry-sdk==1.30.0 opentelemetry-exporter-otlp==1.30.0 openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    raise EnvironmentError("OTEL_EXPORTER_OTLP_ENDPOINT not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

provider = TracerProvider(resource=Resource.create({"service.name": "agentic-ai-observer"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class TraceRecord(BaseModel):
    thread_id: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    had_error: bool


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def load_runbook(runbook_path: Path) -> str | ToolError:
    try:
        return runbook_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        return ToolError(tool_name="load_runbook", error_type="FileNotFoundError", message=str(exc))


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def traced_completion(thread_id: str, prompt: str) -> tuple[str, TraceRecord]:
    enforce_token_budget(6000, prompt)
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("thread_id", thread_id)
        span.set_attribute("model", "gpt-4.1-mini")
        started_at = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
        except (APIError, APITimeoutError, RateLimitError) as exc:
            span.set_attribute("had_error", True)
            log.warning("llm_retry", error_type=type(exc).__name__, message=str(exc))
            raise
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        usage = completion.usage
        span.set_attribute("prompt_tokens", usage.prompt_tokens)
        span.set_attribute("completion_tokens", usage.completion_tokens)
        span.set_attribute("latency_ms", latency_ms)
        return (
            completion.choices[0].message.content or "",
            TraceRecord(
                thread_id=thread_id,
                model="gpt-4.1-mini",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
                had_error=False,
            ),
        )


def traced_tool_call(runbook_path: Path) -> str | ToolError:
    with tracer.start_as_current_span("tool_call") as span:
        result = load_runbook(runbook_path)
        if isinstance(result, ToolError):
            span.set_attribute("had_error", True)
            span.set_attribute("tool.error_type", result.error_type)
            return result
        span.set_attribute("had_error", False)
        span.set_attribute("tool.bytes", len(result))
        return result


if __name__ == "__main__":
    tool_result = traced_tool_call(Path("runbook.md"))
    if isinstance(tool_result, ToolError):
        raise FileNotFoundError(tool_result.message)
    response, trace_record = traced_completion("thread-001", f"Summarize this runbook:\n{tool_result}")
    log.info("trace_complete", response_chars=len(response), trace_record=trace_record.model_dump())
