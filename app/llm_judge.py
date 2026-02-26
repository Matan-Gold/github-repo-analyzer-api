"""Optional LLM judge for eval-mode quality scoring."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import APITimeoutError, OpenAI

from app import config
from app.models import AppError


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for repository summaries.

You must return valid JSON only with exactly these keys:
- overall: number from 0.0 to 1.0
- hallucination_flags: list of short strings, empty list if none
- scores: object with numeric keys faithful, completeness, structure (all 0.0 to 1.0)
- notes: short string

Scoring guidance:
- Penalize claims not clearly grounded in provided response data.
- Penalize missing major components implied by provided structure.
- Prefer conservative grading over optimistic grading."""


class LLMJudge:
    """Runs a single strict JSON grading pass with one invalid-JSON retry."""

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=config.NEBIUS_BASE_URL,
            api_key=config.get_nebius_api_key(),
            timeout=config.TIMEOUT_LLM_SECONDS,
        )
        self.model = config.EVAL_MODEL

    @staticmethod
    def _extract_text(response: Any) -> str:
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

    def _call_json(self, user_prompt: str) -> dict[str, Any]:
        reminder = "\n\nREMINDER: Return valid JSON only with the exact required keys."
        last_error: Exception | None = None
        last_raw = ""

        def _parse_json_payload(raw_text: str) -> dict[str, Any]:
            # Judge models may return JSON in markdown fences; normalize before parsing.
            candidates: list[str] = [raw_text.strip()]

            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL)
            if fenced:
                candidates.append(fenced.group(1).strip())

            # Extract the largest object-like span as a last parse candidate.
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start >= 0 and end > start:
                candidates.append(raw_text[start : end + 1].strip())

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
            raise json.JSONDecodeError("No valid JSON object found in judge output", raw_text, 0)

        for attempt in range(2):
            prompt = user_prompt if attempt == 0 else user_prompt + reminder
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=500,
                )
                raw = self._extract_text(response).strip()
                last_raw = raw
                return _parse_json_payload(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            except APITimeoutError as exc:
                raise AppError(
                    code="LLM_TIMEOUT",
                    message="LLM judge request timed out.",
                    status_code=504,
                ) from exc
            except AppError:
                raise
            except Exception as exc:
                raise AppError(
                    code="INTERNAL_ERROR",
                    message="Unexpected LLM judge provider error.",
                    status_code=500,
                    details={"error": str(exc)},
                ) from exc

        raise AppError(
            code="LLM_INVALID_RESPONSE",
            message="LLM judge returned invalid JSON after retry.",
            status_code=502,
            details={
                "error": str(last_error) if last_error else "invalid_json",
                "raw_preview": (last_raw[:300] if last_raw else ""),
            },
        )

    def judge_summary(self, github_url: str, summarize_payload: dict[str, Any]) -> dict[str, Any]:
        """Return normalized judge output with strict key/type checks."""
        prompt = (
            "Evaluate this repository summary payload for faithfulness and quality.\n"
            f"Repository URL: {github_url}\n\n"
            f"Payload JSON:\n{json.dumps(summarize_payload, ensure_ascii=True)}\n"
        )
        data = self._call_json(prompt)

        required = {"overall", "hallucination_flags", "scores", "notes"}
        if set(data.keys()) != required:
            raise AppError(
                code="LLM_INVALID_RESPONSE",
                message="LLM judge response keys do not match required schema.",
                status_code=502,
                details={"keys": list(data.keys())},
            )

        try:
            overall = float(data["overall"])
        except Exception as exc:
            raise AppError(
                code="LLM_INVALID_RESPONSE",
                message="LLM judge response has invalid overall score.",
                status_code=502,
            ) from exc

        hallucination_flags = data.get("hallucination_flags")
        if not isinstance(hallucination_flags, list) or not all(isinstance(item, str) for item in hallucination_flags):
            raise AppError(
                code="LLM_INVALID_RESPONSE",
                message="LLM judge response has invalid hallucination flags.",
                status_code=502,
            )

        scores = data.get("scores")
        if not isinstance(scores, dict):
            raise AppError(
                code="LLM_INVALID_RESPONSE",
                message="LLM judge response has invalid score details.",
                status_code=502,
            )

        notes = str(data.get("notes", "")).strip()

        return {
            "overall": max(0.0, min(1.0, overall)),
            "hallucination_flags": [item.strip() for item in hallucination_flags if item.strip()],
            "scores": scores,
            "notes": notes,
        }
