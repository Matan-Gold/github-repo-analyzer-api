"""API models and typed application errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    github_url: str = Field(..., examples=["https://github.com/psf/requests"])


class SummarizeResponse(BaseModel):
    summary: str
    technologies: list[str]
    structure: list[str]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class RepoTreeItem(BaseModel):
    path: str
    size: int
    type: str


class SelectedFile(BaseModel):
    path: str
    content: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    source: str = "raw"


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int = 500
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> ErrorResponse:
        # Shared helper for consistent API error serialization.
        return ErrorResponse(error=ErrorBody(code=self.code, message=self.message, details=self.details))
