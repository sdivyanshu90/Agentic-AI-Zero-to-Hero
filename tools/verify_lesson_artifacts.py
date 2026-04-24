#!/usr/bin/env python3
"""
Verify all 25 Agentic-AI lesson artifacts without requiring an OpenAI API key.

Checks performed for every module:
  1. Markdown file exists and contains all 6 required sections.
  2. All four Python artifacts exist.
  3. Every Python artifact passes py_compile (syntax check).
  4. test_eval.py can be collected by pytest (no import-time crash).

Run from the repository root:
    python3 tools/verify_lesson_artifacts.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

MODULES = [
    ("phase-1", "1.1-strict-data-extractor"),
    ("phase-1", "1.2-cli-os-assistant"),
    ("phase-1", "1.3-naive-rag-support-bot"),
    ("phase-1", "1.4-sql-query-generator"),
    ("phase-1", "1.5-intent-router"),
    ("phase-2", "2.1-episodic-memory-logger"),
    ("phase-2", "2.2-graphrag-researcher"),
    ("phase-2", "2.3-context-window-manager"),
    ("phase-2", "2.4-cross-encoder-reranker"),
    ("phase-2", "2.5-agentic-web-scraper"),
    ("phase-3", "3.1-langgraph-basics"),
    ("phase-3", "3.2-reflection-loop-agent"),
    ("phase-3", "3.3-human-in-the-loop-escalation"),
    ("phase-3", "3.4-supervisor-handoff-router"),
    ("phase-3", "3.5-multi-agent-pr-reviewer"),
    ("phase-4", "4.1-llm-as-a-judge-eval-framework"),
    ("phase-4", "4.2-prompt-injection-firewall"),
    ("phase-4", "4.3-tool-execution-sandbox"),
    ("phase-4", "4.4-data-exfiltration-defense"),
    ("phase-4", "4.5-tracing-and-observability"),
    ("phase-5", "5.1-async-task-queue"),
    ("phase-5", "5.2-streaming-tool-ux"),
    ("phase-5", "5.3-fleet-scaling"),
    ("phase-5", "5.4-model-fallback-router"),
    ("phase-5", "5.5-edge-quantization"),
]

REQUIRED_SECTIONS = [
    "## Section 1",
    "## Section 2",
    "## Section 3",
    "## Section 4",
    "## Section 5",
    "## Section 6",
]

PYTHON_ARTIFACTS = [
    "architecture_types.py",
    "from_scratch.py",
    "production.py",
    "test_eval.py",
]


def check_markdown(phase: str, module: str) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    failures: list[str] = []
    md_path = ROOT / phase / f"{module}.md"
    if not md_path.exists():
        failures.append(f"MISSING markdown: {md_path.relative_to(ROOT)}")
        return failures
    text = md_path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        if section not in text:
            failures.append(f"MISSING section '{section}' in {md_path.relative_to(ROOT)}")
    if "ATTACK:" not in text:
        failures.append(f"MISSING 'ATTACK:' block in {md_path.relative_to(ROOT)}")
    if "FAILURE:" not in text:
        failures.append(f"MISSING 'FAILURE:' bullet in {md_path.relative_to(ROOT)}")
    return failures


def check_python_syntax(phase: str, module: str) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    failures: list[str] = []
    dir_path = ROOT / phase / module
    for artifact in PYTHON_ARTIFACTS:
        file_path = dir_path / artifact
        if not file_path.exists():
            failures.append(f"MISSING artifact: {file_path.relative_to(ROOT)}")
            continue
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(
                f"SYNTAX ERROR in {file_path.relative_to(ROOT)}: {result.stderr.strip()}"
            )
    return failures


def check_pytest_collection(phase: str, module: str) -> list[str]:
    """Return a list of failure strings (empty = pass)."""
    test_path = ROOT / phase / module / "test_eval.py"
    if not test_path.exists():
        return [f"MISSING test_eval.py: {test_path.relative_to(ROOT)}"]
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "--collect-only",
            "--import-mode=importlib",
            "-q",
            str(test_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0 and "error" in result.stderr.lower():
        return [
            f"COLLECTION ERROR in {test_path.relative_to(ROOT)}: "
            + result.stderr.strip()[:200]
        ]
    return []


def main() -> None:
    total_failures: list[str] = []
    print(f"{'Module':<45} {'Markdown':>8} {'Syntax':>8} {'Collect':>8}")
    print("-" * 75)

    for phase, module in MODULES:
        md_failures = check_markdown(phase, module)
        syntax_failures = check_python_syntax(phase, module)
        collect_failures = check_pytest_collection(phase, module)

        md_ok = "PASS" if not md_failures else "FAIL"
        syntax_ok = "PASS" if not syntax_failures else "FAIL"
        collect_ok = "PASS" if not collect_failures else "FAIL"

        label = f"{phase}/{module}"
        print(f"{label:<45} {md_ok:>8} {syntax_ok:>8} {collect_ok:>8}")

        total_failures.extend(md_failures + syntax_failures + collect_failures)

    print("-" * 75)
    if total_failures:
        print(f"\n{len(total_failures)} failure(s) found:\n")
        for failure in total_failures:
            print(f"  • {failure}")
        sys.exit(1)
    else:
        print(f"\nAll {len(MODULES)} modules passed all checks.")
        sys.exit(0)


if __name__ == "__main__":
    main()
