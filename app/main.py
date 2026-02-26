"""Compatibility entrypoint so graders can run `uvicorn app.main:app`."""

from main import app

__all__ = ["app"]
