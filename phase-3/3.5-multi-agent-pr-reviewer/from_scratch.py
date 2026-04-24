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


class ReviewFinding(BaseModel):
    category: Literal["correctness", "security", "tests", "maintainability"]
    severity: Literal["low", "medium", "high"]
    finding: str


REVIEW_PROMPTS: dict[str, str] = {
    "correctness": "Review this diff for behavioral regressions and logic bugs. Return JSON array of findings.",
    "security": "Review this diff for secrets, unsafe auth changes, injection, and sandbox escapes. Return JSON array of findings.",
    "tests": "Review this diff for missing tests and weak assertions. Return JSON array of findings.",
}


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def load_diff(diff_path: Path) -> str | ToolError:
    try:
        return diff_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        return ToolError(tool_name="load_diff", error_type="FileNotFoundError", message=str(exc))


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def review_diff(diff_text: str, category: str) -> list[ReviewFinding]:
    prompt = REVIEW_PROMPTS[category]
    enforce_token_budget(7000, prompt, diff_text)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"diff": diff_text[:12000]}, ensure_ascii=True)},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("review_retry", category=category, error_type=type(exc).__name__, message=str(exc))
        raise
    payload = json.loads(completion.choices[0].message.content or "{}")
    return [ReviewFinding.model_validate(item) for item in payload.get("findings", [])]


def merge_findings(*finding_lists: list[ReviewFinding]) -> str:
    flattened = [finding for finding_list in finding_lists for finding in finding_list]
    flattened = sorted(flattened, key=lambda finding: (finding.severity != "high", finding.category, finding.finding))
    return "\n".join(f"[{finding.severity}] {finding.category}: {finding.finding}" for finding in flattened)


if __name__ == "__main__":
    diff_result = load_diff(Path("pull_request.diff"))
    if isinstance(diff_result, ToolError):
        raise FileNotFoundError(diff_result.message)
    correctness = review_diff(diff_result, "correctness")
    security = review_diff(diff_result, "security")
    tests = review_diff(diff_result, "tests")
    log.info("pr_review_complete", merged_review=merge_findings(correctness, security, tests))
