# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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


class QueryGenerationFailed(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class SQLResult(BaseModel):
    query: str
    rows: list[dict[str, object]] = Field(default_factory=list)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def dump_schema(connection: sqlite3.Connection) -> str:
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    parts: list[str] = []
    for table_name in tables:
        pragma_rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        parts.append(f"TABLE {table_name}: {pragma_rows}")
    return "\n".join(parts)


def execute_sql(connection: sqlite3.Connection, query: str) -> SQLResult | ToolError:
    if not query.strip().lower().startswith("select"):
        return ToolError(tool_name="execute_sql", error_type="PermissionError", message="only SELECT queries are allowed")
    try:
        cursor = connection.execute(query)
        column_names = [column[0] for column in cursor.description or []]
        rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        return SQLResult(query=query, rows=rows)
    except sqlite3.OperationalError as exc:
        return ToolError(tool_name="execute_sql", error_type="OperationalError", message=str(exc))


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_sql(question: str, schema_text: str, error_text: str = "") -> str:
    prompt = f"Schema:\n{schema_text}\nQuestion:\n{question}\nPrevious error:\n{error_text}\nReturn SQL only."
    enforce_token_budget(6000, prompt)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "system", "content": "Generate a read-only SQLite SELECT query."}, {"role": "user", "content": prompt}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("sql_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return (completion.choices[0].message.content or "").strip().strip("`")


def answer_question(database_path: Path, question: str) -> SQLResult:
    connection = sqlite3.connect(database_path)
    schema_text = dump_schema(connection)
    last_error = ""
    for retry_count in range(3):
        query = generate_sql(question, schema_text, last_error)
        execution_result = execute_sql(connection, query)
        if not isinstance(execution_result, ToolError):
            return execution_result
        last_error = f"Query: {query}\nError: {execution_result.message}"
        log.warning("sql_repair", retry_count=retry_count, error=execution_result.message)
    raise QueryGenerationFailed(last_error)


if __name__ == "__main__":
    result = answer_question(Path("app.db"), "Which orders shipped this week?")
    log.info("sql_generation_complete", rows=len(result.rows), query=result.query)
