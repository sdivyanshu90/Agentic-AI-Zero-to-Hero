# requirements: pydanticai==0.0.24 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
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


class ExtractionState(BaseModel):
    document_text: str
    retry_count: int = Field(ge=0, le=3, default=0)
    status: Literal["running", "corrected", "failed"] = "running"


class InvoiceExtraction(BaseModel):
    invoice_id: str
    customer_email: EmailStr
    amount_usd: float
    status: Literal["PAID", "PENDING", "REFUNDED"]


class ExtractionStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS extraction_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    def save(self, state: ExtractionState) -> None:
        self._connection.execute("INSERT INTO extraction_runs (payload) VALUES (?)", (state.model_dump_json(),))
        self._connection.commit()


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


extractor = Agent(
    model="openai:gpt-4.1-mini",
    result_type=InvoiceExtraction,
    retries=3,
    system_prompt="Extract the invoice into the exact result schema. Return enum values exactly as declared.",
)


def extract_with_state(document_text: str) -> InvoiceExtraction:
    enforce_token_budget(6000, document_text)
    state = ExtractionState(document_text=document_text)
    store = ExtractionStore(Path("extraction_runs.sqlite3"))
    store.save(state)
    try:
        result = extractor.run_sync(document_text)
        state.status = "corrected" if state.retry_count > 0 else "running"
        store.save(state)
        return result.output
    except Exception as exc:
        state.retry_count = 3
        state.status = "failed"
        store.save(state)
        raise RuntimeError(str(exc)) from exc