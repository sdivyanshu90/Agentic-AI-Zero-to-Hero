# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import hashlib
import os
import uuid
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


class ApprovalRecord(BaseModel):
    thread_id: str
    action_hash: str
    proposed_action: str
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "edit", "reject", "pending"] = "pending"
    edited_action: str = ""
    reviewer_id: str = ""


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def action_hash(thread_id: str, proposed_action: str) -> str:
    return hashlib.sha256(f"{thread_id}:{proposed_action}".encode("utf-8")).hexdigest()


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def plan_action(user_request: str) -> str:
    enforce_token_budget(6000, user_request)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": f"Propose the next action only: {user_request}"}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("plan_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


class ApprovalQueue:
    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}

    def submit(self, proposed_action: str, risk_level: Literal["low", "medium", "high"]) -> ApprovalRecord:
        thread_id = str(uuid.uuid4())
        record = ApprovalRecord(
            thread_id=thread_id,
            action_hash=action_hash(thread_id, proposed_action),
            proposed_action=proposed_action,
            risk_level=risk_level,
        )
        self._records[thread_id] = record
        return record

    def decide(self, thread_id: str, decision: Literal["approve", "edit", "reject"], reviewer_id: str, edited_action: str = "") -> ApprovalRecord:
        record = self._records[thread_id]
        record.decision = decision
        record.reviewer_id = reviewer_id
        record.edited_action = edited_action
        return record


def maybe_execute(record: ApprovalRecord) -> str:
    if record.risk_level == "high" and record.decision == "pending":
        raise PermissionError("high-risk action requires approval")
    if record.decision == "reject":
        return "action rejected"
    return record.edited_action or record.proposed_action


if __name__ == "__main__":
    queue = ApprovalQueue()
    action = plan_action("Refund the enterprise customer and disable the account.")
    record = queue.submit(action, risk_level="high")
    log.info("approval_pending", record=record.model_dump())
