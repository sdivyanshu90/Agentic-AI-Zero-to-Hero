# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BudgetExceededError(RuntimeError):
    pass


class JudgeFailureError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class EvalCase(BaseModel):
    input: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class JudgeScore(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    tone: float = Field(ge=0.0, le=1.0)
    weighted_mean: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class EvalCheckpoint(BaseModel):
    run_id: str
    case_index: int
    status: Literal["pending", "running", "failed", "passed"]
    weighted_mean: float = Field(ge=0.0, le=1.0)


JUDGE_PROMPT: str = """
You are an evaluation judge. Compare the candidate output to the expected output.
Score precision, recall, faithfulness, and tone from 0.0 to 1.0.
Return valid JSON with keys: precision, recall, faithfulness, tone, weighted_mean, reason.
weighted_mean = 0.35*precision + 0.25*recall + 0.30*faithfulness + 0.10*tone.
Do not include markdown.
""".strip()


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def load_cases_from_disk(dataset_path: Path) -> list[EvalCase] | ToolError:
    try:
        raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return ToolError(tool_name="load_cases_from_disk", error_type="FileNotFoundError", message=str(exc))
    except json.JSONDecodeError as exc:
        return ToolError(tool_name="load_cases_from_disk", error_type="JSONDecodeError", message=str(exc))

    validated_cases: list[EvalCase] = []
    for raw_case in raw_cases:
        validated_cases.append(EvalCase.model_validate(raw_case))
    return validated_cases


def init_checkpoint_store(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_checkpoint (
            run_id TEXT NOT NULL,
            case_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            weighted_mean REAL NOT NULL,
            PRIMARY KEY (run_id, case_index)
        )
        """
    )
    connection.commit()
    return connection


def persist_checkpoint(connection: sqlite3.Connection, checkpoint: EvalCheckpoint) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO eval_checkpoint (run_id, case_index, status, weighted_mean) VALUES (?, ?, ?, ?)",
        (checkpoint.run_id, checkpoint.case_index, checkpoint.status, checkpoint.weighted_mean),
    )
    connection.commit()


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def run_candidate(candidate_model: str, user_input: str, max_prompt_tokens: int) -> str:
    enforce_token_budget(max_prompt_tokens, user_input)
    try:
        completion = client.chat.completions.create(
            model=candidate_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Answer the user request directly and accurately."},
                {"role": "user", "content": user_input},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("candidate_call_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    candidate_output = completion.choices[0].message.content or ""
    log.info("candidate_call_complete", candidate_model=candidate_model, output_chars=len(candidate_output))
    return candidate_output


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def run_judge(
    judge_model: str,
    case: EvalCase,
    candidate_output: str,
    max_prompt_tokens: int,
) -> JudgeScore:
    enforce_token_budget(max_prompt_tokens, JUDGE_PROMPT, case.input, case.expected, candidate_output)
    judge_input = json.dumps(
        {
            "input": case.input,
            "expected": case.expected,
            "candidate_output": candidate_output,
        },
        ensure_ascii=True,
    )
    try:
        completion = client.chat.completions.create(
            model=judge_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": judge_input},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("judge_call_retry", error_type=type(exc).__name__, message=str(exc))
        raise

    judge_response_raw = completion.choices[0].message.content or "{}"
    try:
        return JudgeScore.model_validate_json(judge_response_raw)
    except ValidationError as exc:
        raise JudgeFailureError(str(exc)) from exc


def evaluate_dataset(
    dataset_path: Path,
    database_path: Path,
    candidate_model: str,
    judge_model: str,
    threshold: float,
    max_prompt_tokens: int,
) -> float:
    load_result = load_cases_from_disk(dataset_path)
    if isinstance(load_result, ToolError):
        raise FileNotFoundError(load_result.message)

    connection = init_checkpoint_store(database_path)
    run_id = str(uuid.uuid4())
    weighted_means: list[float] = []

    for case_index, case in enumerate(load_result):
        persist_checkpoint(
            connection,
            EvalCheckpoint(run_id=run_id, case_index=case_index, status="running", weighted_mean=0.0),
        )
        candidate_output = run_candidate(candidate_model, case.input, max_prompt_tokens)
        judge_score = run_judge(judge_model, case, candidate_output, max_prompt_tokens)
        weighted_means.append(judge_score.weighted_mean)
        status: Literal["failed", "passed"] = "passed" if judge_score.weighted_mean >= threshold else "failed"
        persist_checkpoint(
            connection,
            EvalCheckpoint(
                run_id=run_id,
                case_index=case_index,
                status=status,
                weighted_mean=judge_score.weighted_mean,
            ),
        )
        log.info(
            "eval_case_complete",
            run_id=run_id,
            case_index=case_index,
            weighted_mean=judge_score.weighted_mean,
            status=status,
        )

    aggregate_weighted_mean = sum(weighted_means) / max(len(weighted_means), 1)
    if aggregate_weighted_mean < threshold:
        raise JudgeFailureError(
            f"aggregate weighted mean {aggregate_weighted_mean:.3f} below threshold {threshold:.3f}"
        )
    return aggregate_weighted_mean


if __name__ == "__main__":
    evaluate_dataset(
        dataset_path=Path("golden/llm_as_a_judge_eval_framework.json"),
        database_path=Path("eval_checkpoint.sqlite3"),
        candidate_model="gpt-4o-mini",
        judge_model="gpt-4.1-mini",
        threshold=0.85,
        max_prompt_tokens=6000,
    )
