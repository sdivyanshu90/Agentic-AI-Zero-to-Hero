# requirements: pydantic==2.11.3 tenacity==8.5.0 structlog==24.4.0 python-dotenv==1.0.1 openai==1.77.0
from __future__ import annotations

import base64
import binascii
import codecs
import os
import re
from dataclasses import dataclass
from pathlib import Path

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set")

structlog.configure(processors=[structlog.processors.JSONRenderer()])
log = structlog.get_logger(__name__)


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolError:
    tool_name: str
    error_type: str
    message: str


class RedactionEvent(BaseModel):
    pattern_name: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class FilterResult(BaseModel):
    filtered_response: str
    redactions: list[RedactionEvent] = Field(default_factory=list)
    blocked: bool = False
    verdict: str


PATTERNS: dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "api_key_openai": r"\bsk-[A-Za-z0-9]{20,}\b",
    "api_key_aws": r"\bAKIA[0-9A-Z]{16}\b",
    "api_key_github": r"\bghp_[A-Za-z0-9]{36}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "ipv4": r"\b(?:10|127|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)(?:\.\d{1,3}){2}\b",
    "base64_blob": r"[A-Za-z0-9+/]{40,}={0,2}",
}


def load_confidential_corpus(corpus_path: Path) -> list[str] | ToolError:
    try:
        return [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError as exc:
        return ToolError(tool_name="load_confidential_corpus", error_type="FileNotFoundError", message=str(exc))


def decode_common_encodings(response_text: str) -> list[str]:
    variants = [response_text]
    try:
        variants.append(codecs.decode(response_text, "rot_13"))
    except UnicodeDecodeError:
        pass
    for candidate in re.findall(PATTERNS["base64_blob"], response_text):
        try:
            decoded_bytes = base64.b64decode(candidate, validate=True)
            variants.append(decoded_bytes.decode("utf-8", errors="ignore"))
        except (binascii.Error, ValueError):
            continue
    return variants


class RegexFilter:
    def apply(self, response_text: str) -> FilterResult:
        filtered_response = response_text
        redactions: list[RedactionEvent] = []
        for pattern_name, pattern in PATTERNS.items():
            for match in re.finditer(pattern, filtered_response):
                redactions.append(
                    RedactionEvent(pattern_name=pattern_name, start_offset=match.start(), end_offset=match.end())
                )
            filtered_response = re.sub(pattern, f"[REDACTED_{pattern_name.upper()}]", filtered_response)
        verdict = "redacted" if redactions else "safe"
        return FilterResult(filtered_response=filtered_response, redactions=redactions, verdict=verdict)


class VerbatimFilter:
    def __init__(self, confidential_corpus: list[str]) -> None:
        self._confidential_corpus = confidential_corpus

    @staticmethod
    def _jaccard_similarity(left_text: str, right_text: str) -> float:
        left_tokens = set(left_text.split())
        right_tokens = set(right_text.split())
        if not left_tokens and not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def is_verbatim_leak(self, response_text: str) -> bool:
        for candidate_text in decode_common_encodings(response_text):
            for start_offset in range(0, max(len(candidate_text) - 511, 1), 256):
                window = candidate_text[start_offset : start_offset + 512]
                for corpus_entry in self._confidential_corpus:
                    if self._jaccard_similarity(window, corpus_entry[:512]) > 0.6:
                        return True
        return False


def filter_response(response_text: str, confidential_corpus: list[str]) -> FilterResult:
    regex_result = RegexFilter().apply(response_text)
    if VerbatimFilter(confidential_corpus).is_verbatim_leak(response_text):
        regex_result.blocked = True
        regex_result.verdict = "blocked"
    log.info("filter_complete", blocked=regex_result.blocked, redaction_count=len(regex_result.redactions))
    return regex_result


if __name__ == "__main__":
    corpus_result = load_confidential_corpus(Path("confidential.txt"))
    if isinstance(corpus_result, ToolError):
        raise FileNotFoundError(corpus_result.message)
    result = filter_response("Customer SSN is 123-45-6789", corpus_result)
    log.info("filter_result", result=result.model_dump())
