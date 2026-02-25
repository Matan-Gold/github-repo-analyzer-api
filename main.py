"""FastAPI entrypoint exposing POST /summarize."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models import AppError, ErrorBody, ErrorResponse, SummarizeRequest, SummarizeResponse
from summarizer import RepositorySummarizer


app = FastAPI(title="GitHub Repo Analyzer API", version="1.0.0")


def _error_response(err: AppError) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=err.code, message=err.message, details=err.details)).model_dump()
    return JSONResponse(status_code=err.status_code, content=payload)


@app.exception_handler(RequestValidationError)
def request_validation_exception_handler(_request, exc: RequestValidationError):
    # Keep validation failures in the same error envelope as runtime errors.
    return _error_response(
        AppError(
            code="INVALID_GITHUB_URL",
            message="Invalid request payload.",
            status_code=422,
            details={"errors": exc.errors()},
        )
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    try:
        orchestrator = RepositorySummarizer()
        result = orchestrator.summarize_repository(request.github_url)
        return result
    except AppError as err:
        return _error_response(err)
    except Exception as exc:  # Safety net preserving error contract.
        return _error_response(
            AppError(
                code="INTERNAL_ERROR",
                message="Unhandled server error.",
                status_code=500,
                details={"error": str(exc)},
            )
        )
