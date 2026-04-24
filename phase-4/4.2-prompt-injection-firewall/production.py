# requirements: fastapi==0.115.12 redis==5.2.1 httpx==0.28.1 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1 openai==1.77.0
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import httpx
import redis.asyncio as redis
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("LLAMAGUARD_URL"):
    raise EnvironmentError("LLAMAGUARD_URL not set")
if not os.getenv("REDIS_URL"):
    raise EnvironmentError("REDIS_URL not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class FirewallState(BaseModel):
    request_id: str
    user_id: str
    normalized_message: str
    decision: Literal["allow", "block", "review"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def normalize_message(raw_message: str) -> str:
    normalized_message = unicodedata.normalize("NFKC", raw_message)
    normalized_message = normalized_message.replace("\u200b", "")
    normalized_message = re.sub(r"\s+", " ", normalized_message.casefold()).strip()
    return normalized_message


def regex_decision(message: str) -> FirewallState:
    normalized_message = normalize_message(message)
    if re.search(r"ignore\s+all\s+previous\s+instructions", normalized_message):
        return FirewallState(
            request_id="pending",
            user_id="pending",
            normalized_message=normalized_message,
            decision="block",
            confidence=0.99,
            reason="matched override payload after unicode normalization",
        )
    return FirewallState(
        request_id="pending",
        user_id="pending",
        normalized_message=normalized_message,
        decision="review",
        confidence=0.50,
        reason="needs classifier",
    )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
async def classify_with_llamaguard(message: str) -> FirewallState:
    enforce_token_budget(4000, message)
    async with httpx.AsyncClient(timeout=10.0) as http_client:
        response = await http_client.post(
            os.getenv("LLAMAGUARD_URL"),
            json={"message": message},
        )
        response.raise_for_status()
    payload = response.json()
    return FirewallState(
        request_id="pending",
        user_id="pending",
        normalized_message=normalize_message(message),
        decision=payload["decision"],
        confidence=float(payload["confidence"]),
        reason=str(payload["reason"]),
    )


async def save_state(redis_key: str, state: FirewallState) -> None:
    await redis_client.set(redis_key, state.model_dump_json(), ex=86400)


app = FastAPI()


@app.middleware("http")
async def prompt_injection_firewall(request: Request, call_next: object) -> JSONResponse:
    if request.url.path != "/chat":
        return await call_next(request)

    payload = await request.json()
    user_message = str(payload.get("message", ""))
    request_id = str(payload.get("request_id", "unknown"))
    user_id = str(payload.get("user_id", "anonymous"))

    state = regex_decision(user_message)
    state.request_id = request_id
    state.user_id = user_id

    if state.decision != "block":
        state = await classify_with_llamaguard(user_message)
        state.request_id = request_id
        state.user_id = user_id

    await save_state(f"firewall:{request_id}", state)
    if state.decision == "block":
        log.warning("request_blocked", request_id=request_id, user_id=user_id, reason=state.reason)
        return JSONResponse(
            status_code=400,
            content={"error": "prompt_injection_detected", "request_id": request_id, "reason": state.reason},
        )
    return await call_next(request)
