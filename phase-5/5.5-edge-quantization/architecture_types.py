from __future__ import annotations

from typing import Literal, TypedDict


class BenchmarkRequest(TypedDict):
    prompt: str
    use_local_model: bool


class BenchmarkResult(TypedDict):
    provider: Literal["local", "remote"]
    ttft_ms: int
    tokens_per_second: float
    quality_score: float


class LocalModelState(TypedDict):
    model_path: str
    gguf_path: str
    server_url: str
    last_result: BenchmarkResult
