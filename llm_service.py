"""Nebius/OpenAI-compatible chat wrapper with strict JSON parsing."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from openai import APITimeoutError, OpenAI

import config
from models import AppError


class LLMService:
    def __init__(self) -> None:
        try:
            api_key = config.get_nebius_api_key()
        except RuntimeError as exc:
            raise AppError(
                code="INTERNAL_ERROR",
                message=str(exc),
                status_code=500,
            ) from exc

        # OpenAI SDK configured against Nebius OpenAI-compatible endpoint.
        self.client = OpenAI(
            base_url=config.NEBIUS_BASE_URL,
            api_key=api_key,
            timeout=config.TIMEOUT_LLM_SECONDS,
        )
        self.model = config.SUMMARIZER_MODEL

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

    def _parse_json_payload(self, raw: str) -> dict[str, Any]:
        # Try raw output first, then common wrappers we see from chat models.
        candidates: list[str] = [raw.strip()]

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1).strip())

        # Fallback extraction for prose that wraps a single JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw[start : end + 1].strip())

        seen: set[str] = set()
        last_exc: json.JSONDecodeError | None = None
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
            if isinstance(parsed, dict):
                return parsed

        if last_exc is not None:
            raise last_exc
        raise json.JSONDecodeError("No valid JSON object found in model output", raw, 0)

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        # One retry for invalid JSON, and one retry for timeout, per spec.
        reminder = "\n\nREMINDER: Return valid JSON only with the exact requested keys."
        last_error: Optional[Exception] = None
        last_raw = ""

        for attempt in range(2):
            prompt = user_prompt if attempt == 0 else user_prompt + reminder
            try:
                raw = self._single_call(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    max_output_tokens=max_output_tokens,
                )
                last_raw = raw
                return self._parse_json_payload(raw)
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
            details={
                "error": str(last_error) if last_error else "invalid_json",
                "raw_preview": (last_raw[:300] if last_raw else ""),
            },
        )
