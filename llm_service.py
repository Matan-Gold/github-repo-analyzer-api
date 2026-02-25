"""Nebius/OpenAI-compatible chat wrapper with strict JSON parsing."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import APITimeoutError, OpenAI

import config
from models import AppError


class LLMService:
    def __init__(self) -> None:
        api_key = os.environ.get(config.NEBIUS_API_KEY_ENV)
        if not api_key:
            raise AppError(
                code="INTERNAL_ERROR",
                message=f"Missing required environment variable: {config.NEBIUS_API_KEY_ENV}",
                status_code=500,
            )

        # OpenAI SDK configured against Nebius OpenAI-compatible endpoint.
        self.client = OpenAI(
            base_url=config.NEBIUS_BASE_URL,
            api_key=api_key,
            timeout=config.TIMEOUT_LLM_SECONDS,
        )
        self.model = config.NEBIUS_MODEL

    def _extract_text(self, response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts)
        return str(content)

    def _single_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=max_output_tokens,
        )
        return self._extract_text(response).strip()

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        # One retry for invalid JSON, and one retry for timeout, per spec.
        reminder = "\n\nREMINDER: Return valid JSON only with the exact requested keys."
        last_error: Exception | None = None

        for attempt in range(2):
            prompt = user_prompt if attempt == 0 else user_prompt + reminder
            try:
                raw = self._single_call(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_output_tokens=max_output_tokens,
                )
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            except APITimeoutError as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise AppError(
                    code="LLM_TIMEOUT",
                    message="LLM request timed out.",
                    status_code=504,
                ) from exc
            except AppError:
                raise
            except Exception as exc:
                raise AppError(
                    code="INTERNAL_ERROR",
                    message="Unexpected LLM provider error.",
                    status_code=500,
                    details={"error": str(exc)},
                ) from exc

        raise AppError(
            code="LLM_INVALID_RESPONSE",
            message="LLM returned invalid JSON after retry.",
            status_code=502,
            details={"error": str(last_error) if last_error else "invalid_json"},
        )
