# test_human_in_the_loop_escalation_eval.py
# requirements: pytest==8.3.5 openai==1.77.0 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import pytest
import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — add it to a .env file to run live tests",
)

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
_api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=_api_key)


class BudgetExceededError(RuntimeError):
    pass


JUDGE_PROMPT: str = """
You are grading a human-approval workflow.
Return JSON with keys score and reason.
score = 1.0 only when risky actions escalate and approvals apply to the exact pending action.
score = 0.0 when execution bypasses approval or stale approvals succeed.
""".strip()


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


@pytest.fixture()
def golden_dataset() -> list[dict[str, Any]]:
    dataset_path = Path("golden/human_in_the_loop_escalation.json")
    if dataset_path.exists():
        return json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        {"input": {"risk": "high", "decision": "approve"}, "expected": "escalated", "tags": ["approve"]},
        {"input": {"risk": "high", "decision": "reject"}, "expected": "escalated", "tags": ["reject"]},
        {"input": {"risk": "medium", "decision": "approve"}, "expected": "escalated", "tags": ["medium"]},
        {"input": {"risk": "low", "decision": "approve"}, "expected": "safe", "tags": ["low"]},
        {"input": {"risk": "high", "decision": "stale"}, "expected": "blocked", "tags": ["stale"]},
    ]


def run_escalation(case_input: dict[str, Any]) -> str:
    if case_input["risk"] == "high" and case_input["decision"] == "stale":
        return "blocked"
    if case_input["risk"] in {"high", "medium"}:
        return "escalated"
    return "safe"


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def judge_score(case_input: dict[str, Any], expected: str, actual: str) -> dict[str, Any]:
    serialized_input = json.dumps(case_input, ensure_ascii=True)
    enforce_token_budget(4000, JUDGE_PROMPT, serialized_input, expected, actual)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": json.dumps({"input": case_input, "expected": expected, "actual": actual}, ensure_ascii=True)},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("judge_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return json.loads(completion.choices[0].message.content or "{}")


def test_eval_threshold(golden_dataset: list[dict[str, Any]]) -> None:
    scored_cases: list[tuple[float, dict[str, Any]]] = []
    for case in golden_dataset:
        actual = run_escalation(dict(case["input"]))
        score_payload = judge_score(dict(case["input"]), str(case["expected"]), actual)
        scored_cases.append((float(score_payload["score"]), score_payload))

    scores = [entry[0] for entry in scored_cases]
    if mean(scores) <= 0.90:
        worst_cases = sorted(zip(golden_dataset, scored_cases), key=lambda item: item[1][0])[:3]
        for case, scored_case in worst_cases:
            log.error("worst_case", input=case["input"], score=scored_case[0], reason=scored_case[1]["reason"])
    assert mean(scores) > 0.90


@pytest.mark.parametrize(
    ("case_input", "expected"),
    [
        ({"risk": "high", "decision": "approve"}, "escalated"),
        ({"risk": "high", "decision": "stale"}, "blocked"),
        ({"risk": "low", "decision": "approve"}, "safe"),
    ],
)
def test_edge_cases(case_input: dict[str, Any], expected: str) -> None:
    actual = run_escalation(case_input)
    scored = judge_score(case_input, expected, actual)
    assert 0.0 <= float(scored["score"]) <= 1.0
    assert isinstance(scored["reason"], str)
