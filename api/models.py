"""
api/models.py
-------------
Pydantic v2 request and response models for the Stock Research API.
All agent output fields are typed as `dict | None` so the API stays
forward-compatible with future agent schema changes.
"""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ── Request ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        examples=["SBIN.NS", "RELIANCE.NS", "AAPL"],
        description="NSE/BSE ticker symbol (e.g. SBIN.NS) or US ticker (e.g. AAPL).",
    )
    depth: Literal["standard", "deep"] | None = Field(
        default="standard",
        description="Analysis depth. 'deep' runs additional sub-analyses.",
    )

    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        forbidden = {";", "&", "|", "`", "$", "(", ")", "<", ">"}
        if any(c in forbidden for c in cleaned):
            raise ValueError(f"Ticker contains forbidden characters: {cleaned!r}")
        return cleaned


# ── Response ──────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    ticker: str
    elapsed_seconds: float = Field(description="Wall-clock time for the full pipeline run.")
    fundamentals_output: dict[str, Any] | None = None
    technical_output:    dict[str, Any] | None = None
    news_output:         dict[str, Any] | None = None
    synthesis_output:    dict[str, Any] | None = None
    critic_output:       dict[str, Any] | None = None


# ── Meta ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str


# ── Report models ─────────────────────────────────────────────────────

class ReportSummary(BaseModel):
    """Lightweight row returned in the history list."""
    id:              str
    ticker:          str
    company_name:    str | None = None
    verdict:         str | None = None
    elapsed_seconds: float | None = None
    created_at:      datetime


class ReportDetail(ReportSummary):
    """Full report including the complete pipeline payload."""
    payload: dict[str, Any]


class SaveReportRequest(BaseModel):
    """
    Body for POST /reports.
    The frontend posts the full AnalyzeResponse JSON here.
    """
    ticker:          str
    elapsed_seconds: float
    company_name:    str | None = None
    verdict:         str | None = None
    fundamentals_output: dict[str, Any] | None = None
    technical_output:    dict[str, Any] | None = None
    news_output:         dict[str, Any] | None = None
    synthesis_output:    dict[str, Any] | None = None
    critic_output:       dict[str, Any] | None = None
