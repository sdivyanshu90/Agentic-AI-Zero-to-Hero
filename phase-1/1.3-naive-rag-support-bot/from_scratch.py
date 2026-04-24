# requirements: openai==1.77.0 sentence-transformers==3.1.1 numpy==2.1.3 pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoder = SentenceTransformer("all-MiniLM-L6-v2")


class BudgetExceededError(RuntimeError):
    pass


class CitationValidationError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(ge=-1.0, le=1.0)


def estimate_prompt_tokens(*parts: str) -> int:
    return sum(max(1, len(part) // 4) for part in parts)


def enforce_token_budget(max_prompt_tokens: int, *parts: str) -> None:
    estimated_prompt_tokens = estimate_prompt_tokens(*parts)
    if estimated_prompt_tokens > max_prompt_tokens:
        raise BudgetExceededError(
            f"estimated prompt tokens {estimated_prompt_tokens} exceed budget {max_prompt_tokens}"
        )


def load_embeddings(embedding_path: Path, chunk_path: Path) -> tuple[np.ndarray, list[dict[str, str]]] | ToolError:
    try:
        return np.load(embedding_path), json.loads(chunk_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return ToolError(tool_name="load_embeddings", error_type="FileNotFoundError", message=str(exc))


def search_docs(query: str, embedding_matrix: np.ndarray, chunks: list[dict[str, str]], top_k: int = 3) -> list[RetrievedChunk]:
    query_vector = encoder.encode([query], normalize_embeddings=True)[0]
    scores = embedding_matrix @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        RetrievedChunk(chunk_id=str(chunks[index]["chunk_id"]), text=str(chunks[index]["text"]), score=float(scores[index]))
        for index in top_indices
    ]


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
def generate_answer(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in retrieved_chunks)
    enforce_token_budget(6000, query, context_block)
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Answer only from retrieved chunks and cite chunk IDs in square brackets."},
                {"role": "user", "content": f"Question: {query}\nRetrieved chunks:\n{context_block}"},
            ],
        )
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("rag_retry", error_type=type(exc).__name__, message=str(exc))
        raise
    return completion.choices[0].message.content or ""


def validate_citations(answer_text: str, retrieved_chunks: list[RetrievedChunk]) -> None:
    cited_chunk_ids = set(re.findall(r"\[([^\]]+)\]", answer_text))
    allowed_chunk_ids = {chunk.chunk_id for chunk in retrieved_chunks}
    if not cited_chunk_ids.issubset(allowed_chunk_ids):
        raise CitationValidationError(f"invalid citations: {sorted(cited_chunk_ids - allowed_chunk_ids)}")


if __name__ == "__main__":
    load_result = load_embeddings(Path("docs/embeddings.npy"), Path("docs/chunks.json"))
    if isinstance(load_result, ToolError):
        raise FileNotFoundError(load_result.message)
    embedding_matrix, chunks = load_result
    retrieved = search_docs("How do I reset MFA after losing my phone?", embedding_matrix, chunks)
    answer = generate_answer("How do I reset MFA after losing my phone?", retrieved)
    validate_citations(answer, retrieved)
    log.info("rag_answer_complete", answer=answer)
