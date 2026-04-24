# requirements: openai==1.77.0 neo4j==5.27.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import structlog
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, Neo4jError
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")
if not os.getenv("NEO4J_URI"):
    raise EnvironmentError("NEO4J_URI not set")
if not os.getenv("NEO4J_USERNAME"):
    raise EnvironmentError("NEO4J_USERNAME not set")
if not os.getenv("NEO4J_PASSWORD"):
    raise EnvironmentError("NEO4J_PASSWORD not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class GraphResult(BaseModel):
    cypher: str
    rows: list[dict[str, object]] = Field(default_factory=list)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def load_schema() -> tuple[str, set[str]]:
    schema_text = "Labels: Company, Investor, Acquisition\nRelationships: INVESTED_IN, ACQUIRED, BUILT"
    return schema_text, {"INVESTED_IN", "ACQUIRED", "BUILT"}


def validate_relationships(cypher: str, allowed_relationships: set[str]) -> None:
    for relationship_name in re.findall(r":([A-Z_]+)\]", cypher):
        if relationship_name not in allowed_relationships:
            raise ValueError(f"invalid relationship type {relationship_name}")


def execute_cypher(cypher: str) -> GraphResult | ToolError:
    try:
        with driver.session() as session:
            rows = [record.data() for record in session.run(cypher)]
        return GraphResult(cypher=cypher, rows=rows)
    except CypherSyntaxError as exc:
        return ToolError(tool_name="execute_cypher", error_type="CypherSyntaxError", message=str(exc))
    except Neo4jError as exc:
        return ToolError(tool_name="execute_cypher", error_type="Neo4jError", message=str(exc))


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_cypher(question: str, schema_text: str, error_text: str = "") -> str:
    prompt = f"Schema:\n{schema_text}\nQuestion:\n{question}\nPrevious error:\n{error_text}\nReturn Cypher only."
    enforce_token_budget(6000, prompt)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[{"role": "system", "content": "Generate Cypher using only valid relationship types from the schema."}, {"role": "user", "content": prompt}],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("cypher_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return (completion.choices[0].message.content or "").strip().strip("`")


def research(question: str) -> GraphResult:
    schema_text, allowed_relationships = load_schema()
    error_text = ""
    for retry_count in range(3):
        cypher = generate_cypher(question, schema_text, error_text)
        try:
            validate_relationships(cypher, allowed_relationships)
        except ValueError as exc:
            error_text = str(exc)
            continue
        execution_result = execute_cypher(cypher)
        if not isinstance(execution_result, ToolError):
            return execution_result
        error_text = execution_result.message
        log.warning("cypher_repair", retry_count=retry_count, error_text=error_text)
    raise RuntimeError(f"graph research failed: {error_text}")


if __name__ == "__main__":
    result = research("Which investors funded the company that acquired the startup that built X?")
    log.info("graph_research_complete", row_count=len(result.rows), cypher=result.cypher)
