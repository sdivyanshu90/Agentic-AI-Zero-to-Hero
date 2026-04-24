# requirements: pydanticai==0.0.24 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)


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
    created_at: str


class MemoryState(BaseModel):
    user_id: str
    user_input: str
    retrieved_fact_count: int = 0


class MemoryStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS memory_facts (user_id TEXT, fact_key TEXT, fact_value TEXT, confidence REAL, created_at TEXT, PRIMARY KEY (user_id, fact_key))"
        )
        self._connection.commit()

    def get(self, user_id: str, max_age_days: int) -> list[MemoryFact]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        rows = self._connection.execute(
            "SELECT fact_key, fact_value, confidence, created_at FROM memory_facts WHERE user_id = ? AND created_at >= ?",
            (user_id, cutoff),
        ).fetchall()
        return [MemoryFact(key=row[0], value=row[1], confidence=float(row[2]), created_at=row[3]) for row in rows]

    def upsert(self, user_id: str, fact: MemoryFact) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO memory_facts (user_id, fact_key, fact_value, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, fact.key, fact.value, fact.confidence, fact.created_at),
        )
        self._connection.commit()


conversation_agent = Agent(model="openai:gpt-4.1-mini", system_prompt="Use known user facts when relevant.")
fact_extractor_agent = Agent(model="openai:gpt-4.1-mini", result_type=list[MemoryFact], system_prompt="Extract durable user facts only.")


def run_turn(user_id: str, user_input: str) -> str:
    store = MemoryStore(Path("episodic_memory.sqlite3"))
    facts = store.get(user_id, max_age_days=30)
    reply = conversation_agent.run_sync(f"Known facts: {facts}\nUser input: {user_input}").output
    extracted_facts = fact_extractor_agent.run_sync(f"User input: {user_input}\nReply: {reply}").output
    for fact in extracted_facts:
        if fact.confidence > 0.8:
            store.upsert(user_id, fact)
    return reply
