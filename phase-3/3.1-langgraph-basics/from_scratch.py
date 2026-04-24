# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
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


class AgentState(BaseModel):
    question: str
    retrieved_context: str = ""
    answer: str = ""
    status: Literal["planned", "retrieved", "answered", "failed"] = "planned"
    retry_count: int = Field(ge=0, default=0)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def retrieve_context(question: str) -> str | ToolError:
    if not question.strip():
        return ToolError(tool_name="retrieve_context", error_type="EmptyQuestion", message="question cannot be empty")
    return f"Context for: {question}"


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def answer_question(question: str, context: str) -> str:
    enforce_token_budget(6000, question, context)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Answer using the retrieved context only when relevant."},
                {"role": "user", "content": f"Question: {question}\nContext: {context}"},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("answer_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


def run_manual_graph(question: str) -> AgentState:
    state = AgentState(question=question)
    tool_result = retrieve_context(question)
    if isinstance(tool_result, ToolError):
        state.status = "failed"
        return state
    state.retrieved_context = tool_result
    state.status = "retrieved"
    state.answer = answer_question(question, state.retrieved_context)
    state.status = "answered"
    return state


if __name__ == "__main__":
    state = run_manual_graph("Why do explicit graph states help debugging?")
    log.info("manual_graph_complete", state=state.model_dump())
