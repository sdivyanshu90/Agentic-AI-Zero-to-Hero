# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
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


class CritiqueResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    critique: str


class ReflectionState(BaseModel):
    task: str
    draft: str = ""
    critique: str = ""
    revised_answer: str = ""
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    retry_count: int = Field(ge=0, default=0)
    status: Literal["drafted", "critiqued", "revised", "done", "failed"] = "drafted"


CRITIC_PROMPT: str = "Return JSON with score and critique. Score the answer from 0.0 to 1.0 and focus on correctness, completeness, and policy compliance."


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def draft_answer(task: str) -> str:
    enforce_token_budget(6000, task)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": task}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("draft_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def critique_answer(task: str, draft: str) -> CritiqueResult:
    enforce_token_budget(6000, CRITIC_PROMPT, task, draft)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": json.dumps({"task": task, "draft": draft}, ensure_ascii=True)},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("critique_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return CritiqueResult.model_validate_json(completion.choices[0].message.content or "{}")


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def revise_answer(task: str, draft: str, critique: str) -> str:
    enforce_token_budget(6000, task, draft, critique)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Revise the answer to fix the critique without adding unsupported claims."},
                {"role": "user", "content": json.dumps({"task": task, "draft": draft, "critique": critique}, ensure_ascii=True)},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("revision_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


def reflect(task: str, score_threshold: float = 0.85, max_retries: int = 2) -> ReflectionState:
    state = ReflectionState(task=task)
    state.draft = draft_answer(task)
    for _ in range(max_retries + 1):
        critique = critique_answer(task, state.draft if not state.revised_answer else state.revised_answer)
        state.critique = critique.critique
        state.score = critique.score
        state.status = "critiqued"
        if critique.score >= score_threshold:
            state.status = "done"
            return state
        state.revised_answer = revise_answer(task, state.draft if not state.revised_answer else state.revised_answer, critique.critique)
        state.retry_count += 1
        state.status = "revised"
    state.status = "failed"
    return state


if __name__ == "__main__":
    state = reflect("Write a rollback checklist for a failed deployment.")
    log.info("reflection_complete", state=state.model_dump())
