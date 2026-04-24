# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass
from typing import Callable

import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel
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


class IntentEnum(str, enum.Enum):
    ORDER_STATUS = "ORDER_STATUS"
    REFUND = "REFUND"
    TECH_SUPPORT = "TECH_SUPPORT"
    UNKNOWN = "UNKNOWN"


class IntentClassification(BaseModel):
    intent: str


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def handle_order_status(user_input: str) -> str:
    return f"order handler: {user_input}"


def handle_refund(user_input: str) -> str:
    return f"refund handler: {user_input}"


def handle_tech_support(user_input: str) -> str:
    return f"tech handler: {user_input}"


def handle_unknown_intent(user_input: str) -> str:
    return f"unknown handler: {user_input}"


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def classify_intent(user_input: str) -> IntentEnum:
    prompt = "Classify the request as ORDER_STATUS, REFUND, TECH_SUPPORT, or UNKNOWN. Return only the enum value name."
    enforce_token_budget(4000, prompt, user_input)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("router_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    raw_intent = (completion.choices[0].message.content or "UNKNOWN").strip().upper()
    try:
        return IntentEnum(raw_intent)
    except ValueError:
        return IntentEnum.UNKNOWN


def route(user_input: str) -> str:
    handlers: dict[IntentEnum, Callable[[str], str]] = {
        IntentEnum.ORDER_STATUS: handle_order_status,
        IntentEnum.REFUND: handle_refund,
        IntentEnum.TECH_SUPPORT: handle_tech_support,
        IntentEnum.UNKNOWN: handle_unknown_intent,
    }
    intent = classify_intent(user_input)
    return handlers[intent](user_input)


if __name__ == "__main__":
    result = route("Where is my order #12345?")
    log.info("router_complete", result=result)
