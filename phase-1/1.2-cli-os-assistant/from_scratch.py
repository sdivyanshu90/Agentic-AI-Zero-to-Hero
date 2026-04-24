# requirements: openai==1.77.0 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", os.getcwd())).resolve()


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class ToolEnvelope(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Run a workspace-local command",
            "parameters": {
                "type": "object",
                "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
                "required": ["argv"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file inside the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file inside the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
]


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def resolve_workspace_path(raw_path: str) -> Path | ToolError:
    resolved_path = (WORKSPACE_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    if not str(resolved_path).startswith(str(WORKSPACE_ROOT)):
        return ToolError(tool_name="path_guard", error_type="PathTraversalError", message=f"rejected path: {raw_path}")
    return resolved_path


def bash_exec(argv: list[str]) -> str | ToolError:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=10, cwd=WORKSPACE_ROOT, shell=False, check=False)
        return completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as exc:
        return ToolError(tool_name="bash_exec", error_type="TimeoutExpired", message=str(exc))
    except FileNotFoundError as exc:
        return ToolError(tool_name="bash_exec", error_type="FileNotFoundError", message=str(exc))


def safe_read_file(raw_path: str) -> str | ToolError:
    resolved_path = resolve_workspace_path(raw_path)
    if isinstance(resolved_path, ToolError):
        return resolved_path
    try:
        return resolved_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        return ToolError(tool_name="read_file", error_type="FileNotFoundError", message=str(exc))


def safe_write_file(raw_path: str, content: str) -> str | ToolError:
    resolved_path = resolve_workspace_path(raw_path)
    if isinstance(resolved_path, ToolError):
        return resolved_path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(content, encoding="utf-8")
    return "write complete"


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def run_agent(messages: list[dict[str, str]]) -> Any:
    serialized_messages = json.dumps(messages, ensure_ascii=True)
    enforce_token_budget(6000, serialized_messages)
    try:
        return client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("agent_retry", error_type=type(exc).__name__, message=str(exc))
        raise


def agent_loop(user_task: str) -> str:
    messages: list[dict[str, str]] = [{"role": "user", "content": user_task}]
    for iteration_count in range(10):
        response = run_agent(messages)
        choice = response.choices[0].message
        if not choice.tool_calls:
            return choice.content or ""
        for tool_call in choice.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "bash_exec":
                tool_result = bash_exec(arguments["argv"])
            elif tool_call.function.name == "read_file":
                tool_result = safe_read_file(arguments["path"])
            else:
                tool_result = safe_write_file(arguments["path"], arguments["content"])
            tool_output = tool_result.message if isinstance(tool_result, ToolError) else tool_result
            messages.append({"role": "tool", "content": tool_output, "tool_call_id": tool_call.id})
        log.info("tool_iteration_complete", iteration_count=iteration_count)
    raise RuntimeError("agent loop exceeded 10 iterations")


if __name__ == "__main__":
    answer = agent_loop("Create notes/summary.txt with the first three files in this directory and then show the content.")
    log.info("cli_assistant_complete", answer=answer)
