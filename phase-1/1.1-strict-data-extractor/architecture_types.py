from __future__ import annotations

from typing import Literal, TypedDict


class ExtractionOutput(TypedDict):
    invoice_id: str
    customer_email: str
    amount_usd: float
    status: Literal["PAID", "PENDING", "REFUNDED"]


class CorrectionAttempt(TypedDict):
    attempt_number: int
    error_text: str
    bad_output: str


class ExtractionState(TypedDict):
    document_text: str
    correction_attempts: list[CorrectionAttempt]
    output: ExtractionOutput
    status: Literal["running", "corrected", "failed"]