# Agentic AI — Zero to Hero

A structured, production-style curriculum for building, hardening, and shipping AI agents. The course covers 25 modules across 5 phases — from the raw primitives every agent is built on (Phase 1) to deployment and runtime concerns that matter at scale (Phase 5).

Every module pairs a detailed markdown lesson with four Python files you can read and run:

| File                    | Purpose                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `architecture_types.py` | Typed data contracts (TypedDicts, Pydantic models) used across the implementation files    |
| `from_scratch.py`       | A single-file, dependency-light implementation you can read top-to-bottom and run directly |
| `production.py`         | A refactored, production-ready version using LangGraph, Pydantic v2, structlog, etc.       |
| `test_eval.py`          | An LLM-as-a-judge eval harness you can run against a live API key                          |

---

## Prerequisites

- **Python 3.10+**
- An **OpenAI API key** (required to run eval harnesses; all other code works offline)
- Optional: **Neo4j** (for module 2.2), **Docker** (for module 4.3), a running **llama.cpp server** (for module 5.5)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/sdivyanshu90/agentic-ai-zero-to-hero.git
cd agentic-ai-zero-to-hero

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install openai>=1.77.0 pydantic[email]>=2.0 structlog langchain-openai langgraph \
            sentence-transformers numpy tiktoken httpx python-dotenv

# 4. (Optional) Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

---

## Running the Code

### Run a single from-scratch implementation

```bash
python3 phase-1/1.1-strict-data-extractor/from_scratch.py
```

### Run the eval harness for one module (requires API key)

```bash
python3 -m pytest phase-1/1.1-strict-data-extractor/test_eval.py -v
```

### Run all eval harnesses (skip gracefully without API key)

```bash
python3 -m pytest phase-1/ phase-2/ phase-3/ phase-4/ phase-5/ -v
```

Without `OPENAI_API_KEY` set, all 100 tests are skipped with a clear message rather than failing.

### Verify all 25 modules pass structural checks (no API key needed)

```bash
python3 tools/verify_lesson_artifacts.py
```

This checks every module for: markdown section structure, Python syntax, and pytest collection. All 25 modules should show `PASS / PASS / PASS`.

---

## Module Structure — How Each Lesson is Organized

Every markdown lesson file follows the same six-section structure:

### Section 1 — Core Intuition

A plain-English explanation of what problem this pattern solves and why it matters. Written to be understood without prior knowledge of the specific tools used.

### Section 2 — Architecture & Information Flow

A walkthrough of `architecture_types.py` — the typed state and payload contracts. Understanding the data shapes before reading the implementation makes every function's purpose immediately clear.

### Section 3 — From-Scratch Implementation

A line-by-line walkthrough of `from_scratch.py`. Each function is explained in terms of what it does, what can go wrong if it were missing, and what design decision it embodies.

### Section 4 — Production Implementation

A walkthrough of `production.py`, which replaces hand-rolled patterns with production-grade libraries (LangGraph state machines, Pydantic validators, structlog, OpenTelemetry). The focus is on what the library adds beyond the from-scratch version.

### Section 5 — Evaluation & Security

Three concrete attack scenarios for each module, each with:

- **ATTACK** — the exact exploit path (what an adversary sends and what goes wrong)
- **MITIGATION** — the specific code that prevents the attack (with snippet)
- **EVAL** — what the LLM judge measures in `test_eval.py`

### Section 6 — Pitfalls & Pro-Tips

Common mistakes, non-obvious trade-offs, and practical advice for adapting the pattern to real production systems.

---

## Phase Overview

### Phase 1 — Agent Primitives

The foundational building blocks every production agent needs before adding memory, orchestration, or multi-agent coordination. Each module isolates one primitive and demonstrates how to harden it.

| Module                                                            | What You Build                               | Core Skill                                        |
| ----------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------- |
| [1.1 Strict Data Extractor](phase-1/1.1-strict-data-extractor.md) | Invoice field extractor with self-correction | Structured output, correction loops               |
| [1.2 CLI OS Assistant](phase-1/1.2-cli-os-assistant.md)           | Sandboxed shell agent                        | Tool calling, path containment, human-in-the-loop |
| [1.3 Naive RAG Support Bot](phase-1/1.3-naive-rag-support-bot.md) | Retrieval-augmented Q&A bot                  | Citation validation, chunk grounding              |
| [1.4 SQL Query Generator](phase-1/1.4-sql-query-generator.md)     | Natural language to SQL                      | Schema grounding, injection prevention            |
| [1.5 Intent Router](phase-1/1.5-intent-router.md)                 | Intent classifier + dispatcher               | Enum normalization, deterministic execution       |

### Phase 2 — Memory Architectures

Five different approaches to giving an agent persistent context. The modules progress from the simplest (key-value episodic store) to the most sophisticated (graph-based multi-hop research).

| Module                                                              | What You Build                             | Core Skill                                 |
| ------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| [2.1 Episodic Memory Logger](phase-2/2.1-episodic-memory-logger.md) | Per-user fact store with confidence gating | Upsert, staleness, poisoning defence       |
| [2.2 GraphRAG Researcher](phase-2/2.2-graphrag-researcher.md)       | Neo4j-backed multi-hop research agent      | Cypher generation, relationship validation |
| [2.3 Context Window Manager](phase-2/2.3-context-window-manager.md) | Priority-aware prompt assembler            | tiktoken counting, summarization           |
| [2.4 Cross-Encoder Reranker](phase-2/2.4-cross-encoder-reranker.md) | Two-stage retrieval with cross-encoder     | Score computation, candidate sanitization  |
| [2.5 Agentic Web Scraper](phase-2/2.5-agentic-web-scraper.md)       | SSRF-safe web research agent               | Domain allowlists, redirect validation     |

### Phase 3 — Stateful Orchestration

Replacing ad-hoc loops with explicit state machines. These modules use LangGraph to build agents whose execution graphs can be inspected, interrupted, resumed, and scaled.

| Module                                                                          | What You Build                      | Core Skill                               |
| ------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------- |
| [3.1 LangGraph Basics](phase-3/3.1-langgraph-basics.md)                         | Minimal LangGraph agent             | Node/edge wiring, state schema           |
| [3.2 Reflection Loop Agent](phase-3/3.2-reflection-loop-agent.md)               | Draft-critique-revise loop          | Bounded retries, separate critic prompt  |
| [3.3 Human-in-the-Loop Escalation](phase-3/3.3-human-in-the-loop-escalation.md) | Risk-gated approval workflow        | Interrupt points, tamper-evident records |
| [3.4 Supervisor Handoff Router](phase-3/3.4-supervisor-handoff-router.md)       | Multi-specialist routing supervisor | Validated route types, handler dispatch  |
| [3.5 Multi-Agent PR Reviewer](phase-3/3.5-multi-agent-pr-reviewer.md)           | Parallel code review agents         | Independent reviewers, diff token budget |

### Phase 4 — Evaluation & Security

Systematic quality assurance and defence-in-depth for production agent systems. Each module addresses a specific attack surface or observability gap.

| Module                                                                            | What You Build                | Core Skill                             |
| --------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------- |
| [4.1 LLM-as-a-Judge Eval Framework](phase-4/4.1-llm-as-a-judge-eval-framework.md) | Four-dimension output scorer  | Rubric design, score validation        |
| [4.2 Prompt Injection Firewall](phase-4/4.2-prompt-injection-firewall.md)         | Multi-pass injection detector | Unicode normalization, regex scanning  |
| [4.3 Tool Execution Sandbox](phase-4/4.3-tool-execution-sandbox.md)               | Docker-isolated code runner   | Resource limits, network isolation     |
| [4.4 Data Exfiltration Defense](phase-4/4.4-data-exfiltration-defense.md)         | Response-layer PII redactor   | Pattern matching, structured redaction |
| [4.5 Tracing and Observability](phase-4/4.5-tracing-and-observability.md)         | OpenTelemetry span wrapper    | Span attributes, batch export          |

### Phase 5 — Deployment & Runtime

Operational concerns that only surface at scale: async throughput, streaming UX, concurrency control, provider resilience, and inference at the edge.

| Module                                                            | What You Build                         | Core Skill                                 |
| ----------------------------------------------------------------- | -------------------------------------- | ------------------------------------------ |
| [5.1 Async Task Queue](phase-5/5.1-async-task-queue.md)           | Bounded multi-worker task queue        | Thread safety, failure isolation           |
| [5.2 Streaming Tool UX](phase-5/5.2-streaming-tool-ux.md)         | Server-sent event stream               | Typed events, sequence IDs                 |
| [5.3 Fleet Scaling](phase-5/5.3-fleet-scaling.md)                 | Concurrency limiter with load shedding | In-flight counter, rejection policy        |
| [5.4 Model Fallback Router](phase-5/5.4-model-fallback-router.md) | OpenAI → Anthropic fallback chain      | Error classification, cascade failure      |
| [5.5 Edge Quantization](phase-5/5.5-edge-quantization.md)         | Local quantized model benchmark        | llama.cpp integration, latency measurement |

---

## Verification Tool

`tools/verify_lesson_artifacts.py` checks all 25 modules without an API key:

```
python3 tools/verify_lesson_artifacts.py

[1.1-strict-data-extractor]  markdown=PASS  syntax=PASS  collect=PASS
[1.2-cli-os-assistant]       markdown=PASS  syntax=PASS  collect=PASS
...
[5.5-edge-quantization]      markdown=PASS  syntax=PASS  collect=PASS

All 25 modules OK.
```

- **markdown** — confirms the file has all six sections plus at least one ATTACK block
- **syntax** — runs `py_compile` on all four Python files in the module directory
- **collect** — runs `pytest --collect-only` and confirms tests can be discovered without errors

---

## Configuration Notes

**pytest** is configured in `pytest.ini` to use `--import-mode=importlib` so all 25 `test_eval.py` files (same name, different directories) can coexist without namespace collisions.

**Environment variables** used across the curriculum:

| Variable                                    | Required by                                         |
| ------------------------------------------- | --------------------------------------------------- |
| `OPENAI_API_KEY`                            | All eval harnesses (tests skip gracefully if unset) |
| `ANTHROPIC_API_KEY`                         | Module 5.4 fallback router                          |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Module 2.2 GraphRAG                                 |
| `LLAMA_SERVER_URL`                          | Module 5.5 edge quantization                        |

## Repository Notes

- Every module includes pinned dependencies in code blocks, a pytest-style eval harness, and explicit security analysis.
- The repository is markdown-first by design, but each lesson now stores its executable code in a sibling directory named after the module file stem.
