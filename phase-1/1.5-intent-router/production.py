# requirements: pydanticai==0.0.24 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import enum
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel
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


class IntentEnum(str, enum.Enum):
    ORDER_STATUS = "ORDER_STATUS"
    REFUND = "REFUND"
    TECH_SUPPORT = "TECH_SUPPORT"
    UNKNOWN = "UNKNOWN"


class IntentClassification(BaseModel):
    intent: IntentEnum


class RouteState(BaseModel):
    user_input: str
    intent: IntentEnum


class RouteStore:
    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("CREATE TABLE IF NOT EXISTS route_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)")
        self._connection.commit()

    def save(self, state: RouteState) -> None:
        self._connection.execute("INSERT INTO route_runs (payload) VALUES (?)", (state.model_dump_json(),))
        self._connection.commit()


classifier = Agent(
    model="openai:gpt-4.1-mini",
    result_type=IntentClassification,
    system_prompt="Return one intent enum only: ORDER_STATUS, REFUND, TECH_SUPPORT, UNKNOWN.",
)


def handle_order_status(user_input: str) -> str:
    return f"order handler: {user_input}"


def handle_refund(user_input: str) -> str:
    return f"refund handler: {user_input}"


def handle_tech_support(user_input: str) -> str:
    return f"tech handler: {user_input}"


def handle_unknown_intent(user_input: str) -> str:
    return f"unknown handler: {user_input}"


def route_request(user_input: str) -> str:
    result = classifier.run_sync(user_input)
    store = RouteStore(Path("intent_routes.sqlite3"))
    state = RouteState(user_input=user_input, intent=result.output.intent)
    store.save(state)
    handlers: dict[IntentEnum, Callable[[str], str]] = {
        IntentEnum.ORDER_STATUS: handle_order_status,
        IntentEnum.REFUND: handle_refund,
        IntentEnum.TECH_SUPPORT: handle_tech_support,
        IntentEnum.UNKNOWN: handle_unknown_intent,
    }
    return handlers[state.intent](user_input)
