# test_tracing_and_observability_eval.py
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
You are grading agent traces.
Return JSON with keys score and reason.
score = 1.0 only when the trace captures llm/tool spans, token counts, latency, and does not expose secrets.
score = 0.0 when spans are missing or sensitive values are logged.
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
    dataset_path = Path("golden/tracing_and_observability.json")
    if dataset_path.exists():
        return json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        {"input": {"spans": ["llm_call", "tool_call"], "tokens": 1200}, "expected": "complete", "tags": ["happy_path"]},
        {"input": {"spans": ["llm_call"], "tokens": 300}, "expected": "incomplete", "tags": ["missing_tool"]},
        {"input": {"spans": ["tool_call"], "tokens": 0}, "expected": "incomplete", "tags": ["missing_llm"]},
        {"input": {"spans": ["llm_call", "tool_call"], "tokens": 22000}, "expected": "alert", "tags": ["high_tokens"]},
        {"input": {"spans": ["llm_call"], "secret": "sk-abc123abc123abc123abc123"}, "expected": "unsafe", "tags": ["secret_leak"]},
    ]


def run_trace_audit(trace_payload: dict[str, Any]) -> str:
    spans = set(trace_payload.get("spans", []))
    if "secret" in trace_payload:
        return "unsafe"
    if {"llm_call", "tool_call"}.issubset(spans) and int(trace_payload.get("tokens", 0)) <= 20000:
        return "complete"
    if int(trace_payload.get("tokens", 0)) > 20000:
        return "alert"
    return "incomplete"


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
        actual = run_trace_audit(dict(case["input"]))
        score_payload = judge_score(dict(case["input"]), str(case["expected"]), actual)
        scored_cases.append((float(score_payload["score"]), score_payload))

    scores = [entry[0] for entry in scored_cases]
    if mean(scores) <= 0.99:
        worst_cases = sorted(zip(golden_dataset, scored_cases), key=lambda item: item[1][0])[:3]
        for case, scored_case in worst_cases:
            log.error("worst_case", input=case["input"], score=scored_case[0], reason=scored_case[1]["reason"])
    assert mean(scores) > 0.99


@pytest.mark.parametrize(
    ("case_input", "expected"),
    [
        ({}, "incomplete"),
        ({"spans": ["llm_call"], "secret": "AKIA1234567890ABCDEF"}, "unsafe"),
        ({"spans": ["llm_call", "tool_call"], "tokens": 500000}, "alert"),
    ],
)
def test_edge_cases(case_input: dict[str, Any], expected: str) -> None:
    actual = run_trace_audit(case_input)
    scored = judge_score(case_input, expected, actual)
    assert 0.0 <= float(scored["score"]) <= 1.0
    assert isinstance(scored["reason"], str)
