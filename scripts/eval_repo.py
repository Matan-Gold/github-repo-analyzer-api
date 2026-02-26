"""Run optional eval-mode judge against local /summarize output."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from llm_judge import LLMJudge
from models import AppError


def _print_mode_instructions() -> None:
    print("Judge execution is disabled for this environment.")
    print("Enable both flags before running:")
    print("  export ENVIRONMENT=eval")
    print("  export ENABLE_JUDGE=1")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/eval_repo.py <github_url>")
        return 2

    github_url = argv[1].strip()
    if not github_url:
        print("github_url cannot be empty")
        return 2

    # Judge is strictly opt-in to keep default app behavior unchanged.
    if not config.IS_EVAL or not config.ENABLE_JUDGE:
        _print_mode_instructions()
        return 2

    try:
        # Large repositories can exceed short local HTTP timeouts.
        summarize_timeout = float(os.getenv("EVAL_SUMMARIZE_TIMEOUT_SECONDS", "300"))
    except ValueError:
        summarize_timeout = 300.0

    base_url = os.getenv("EVAL_BASE_URL", "http://localhost:8000").rstrip("/")
    try:
        response = requests.post(
            f"{base_url}/summarize",
            json={"github_url": github_url},
            timeout=summarize_timeout,
        )
    except requests.RequestException as exc:
        print(f"Failed to call summarize endpoint: {exc}")
        return 1

    if response.status_code != 200:
        print(f"Summarize endpoint returned {response.status_code}")
        try:
            print(json.dumps(response.json(), ensure_ascii=True, indent=2))
        except Exception:
            print(response.text)
        return 1

    summarize_payload = response.json()

    try:
        judge = LLMJudge()
        result = judge.judge_summary(github_url, summarize_payload)
    except AppError as exc:
        print("Judge execution failed:")
        print(
            json.dumps(
                {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 1
    except Exception as exc:
        print(f"Judge execution failed: {repr(exc)}")
        return 1

    overall = float(result.get("overall", 0.0))
    flags = result.get("hallucination_flags", [])

    print(json.dumps(result, ensure_ascii=True, indent=2))

    # Exit code contract: 0 pass, 1 evaluated-but-failed, 2 mode disabled.
    passed = overall >= 0.75 and len(flags) == 0
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
