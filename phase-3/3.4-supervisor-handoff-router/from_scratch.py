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


class HandoffState(BaseModel):
    user_request: str
    route: Literal["billing", "security", "docs", "none"] = "none"
    specialist_response: str = ""
    retry_count: int = Field(ge=0, default=0)
    status: Literal["routed", "handled", "failed"] = "routed"


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def classify_route(user_request: str) -> Literal["billing", "security", "docs", "none"]:
    enforce_token_budget(4000, user_request)
    if any(word in user_request.casefold() for word in ["refund", "invoice", "charge"]):
        return "billing"
    if any(word in user_request.casefold() for word in ["token", "secret", "permission"]):
        return "security"
    if any(word in user_request.casefold() for word in ["docs", "guide", "api"]):
        return "docs"
    return "none"


def billing_agent(user_request: str) -> str:
    return f"Billing specialist handled: {user_request}"


def security_agent(user_request: str) -> str:
    return f"Security specialist handled: {user_request}"


def docs_agent(user_request: str) -> str:
    return f"Docs specialist handled: {user_request}"


def run_supervisor(user_request: str) -> HandoffState:
    state = HandoffState(user_request=user_request)
    state.route = classify_route(user_request)
    if state.route == "billing":
        state.specialist_response = billing_agent(user_request)
    elif state.route == "security":
        state.specialist_response = security_agent(user_request)
    elif state.route == "docs":
        state.specialist_response = docs_agent(user_request)
    else:
        state.status = "failed"
        return state
    state.status = "handled"
    return state


if __name__ == "__main__":
    state = run_supervisor("Refund the duplicated invoice and explain the charge.")
    log.info("handoff_complete", state=state.model_dump())
