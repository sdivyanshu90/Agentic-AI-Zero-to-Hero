# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
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


class MemoryFact(BaseModel):
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


EXTRACTOR_PROMPT: str = "Return JSON array of facts with keys key, value, confidence. Only persist durable user facts."


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


class MemoryStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS memory_facts (user_id TEXT, fact_key TEXT, fact_value TEXT, confidence REAL, PRIMARY KEY (user_id, fact_key))"
        )
        self._connection.commit()

    def get(self, user_id: str) -> list[MemoryFact]:
        rows = self._connection.execute(
            "SELECT fact_key, fact_value, confidence FROM memory_facts WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [MemoryFact(key=row[0], value=row[1], confidence=float(row[2])) for row in rows]

    def upsert(self, user_id: str, fact: MemoryFact) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO memory_facts (user_id, fact_key, fact_value, confidence) VALUES (?, ?, ?, ?)",
            (user_id, fact.key, fact.value, fact.confidence),
        )
        self._connection.commit()


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def chat_with_memory(user_input: str, known_facts: list[MemoryFact]) -> str:
    memory_block = "\n".join(f"- {fact.key}: {fact.value}" for fact in known_facts)
    enforce_token_budget(6000, user_input, memory_block)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": f"Known about this user:\n{memory_block}"},
                {"role": "user", "content": user_input},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("memory_chat_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def extract_facts(user_input: str, assistant_reply: str) -> list[MemoryFact]:
    payload = json.dumps({"user_input": user_input, "assistant_reply": assistant_reply}, ensure_ascii=True)
    enforce_token_budget(4000, EXTRACTOR_PROMPT, payload)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": EXTRACTOR_PROMPT}, {"role": "user", "content": payload}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("fact_extractor_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    raw_payload = json.loads(completion.choices[0].message.content or "{}")
    return [MemoryFact.model_validate(item) for item in raw_payload.get("facts", [])]


if __name__ == "__main__":
    store = MemoryStore(Path("memory.sqlite3"))
    user_id = "user-123"
    response = chat_with_memory("I switched from Python to TypeScript.", store.get(user_id))
    for fact in extract_facts("I switched from Python to TypeScript.", response):
        if fact.confidence > 0.8:
            store.upsert(user_id, fact)
    log.info("memory_turn_complete", stored_facts=[fact.model_dump() for fact in store.get(user_id)])
