# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, EmailStr, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BudgetExceededError(RuntimeError):
    pass


class ExtractionFailedError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class InvoiceExtraction(BaseModel):
    invoice_id: str = Field(min_length=1)
    customer_email: EmailStr
    amount_usd: float
    status: Literal["PAID", "PENDING", "REFUNDED"]


EXTRACTION_PROMPT: str = "Return only valid JSON for the invoice schema."


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def load_document(document_path: Path) -> str | ToolError:
    try:
        return document_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        return ToolError(tool_name="load_document", error_type="FileNotFoundError", message=str(exc))


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def request_extraction(prompt: str, document_text: str) -> str:
    enforce_token_budget(6000, prompt, document_text)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("extract_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or "{}"


def extract_document(document_text: str) -> InvoiceExtraction:
    prompt = EXTRACTION_PROMPT
    for attempt_number in range(1, 4):
        llm_response_raw = request_extraction(prompt, document_text)
        try:
            parsed_payload = json.loads(llm_response_raw)
            return InvoiceExtraction.model_validate(parsed_payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            prompt = (
                f"Schema: {InvoiceExtraction.model_json_schema()}\n"
                f"Bad output: {llm_response_raw}\n"
                f"Validation error: {str(exc)}\n"
                f"Attempt: {attempt_number}/3\n"
                "Return corrected JSON only."
            )
            log.warning("extraction_correction", attempt_number=attempt_number, error_text=str(exc))
    raise ExtractionFailedError("extraction failed after 3 correction attempts")


if __name__ == "__main__":
    document_result = load_document(Path("invoice.txt"))
    if isinstance(document_result, ToolError):
        raise FileNotFoundError(document_result.message)
    extraction = extract_document(document_result)
    log.info("extraction_complete", extraction=extraction.model_dump())