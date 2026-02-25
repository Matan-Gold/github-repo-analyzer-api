"""Application configuration and hard limits."""

from __future__ import annotations

import os


# Nebius Token Factory defaults (override via environment if needed).
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
NEBIUS_API_KEY_ENV = "NEBIUS_API_KEY"

# Pipeline limits and safety budgets.
MAX_SELECTED_FILES = 10
MAX_FILE_BYTES = 200_000
MAX_FILE_TOKENS = 8_000
CHUNK_TOKENS = 2_000
SAFE_CONTEXT = 100_000

TIMEOUT_GITHUB_SECONDS = 10
TIMEOUT_LLM_SECONDS = 60

CONCURRENCY_GITHUB = 5
CONCURRENCY_LLM = 3

TEMPERATURE = 0
MAX_OUTPUT_TOKENS_PLANNER = 500
MAX_OUTPUT_TOKENS_CHUNK = 700
MAX_OUTPUT_TOKENS_FINAL = 1000

# GitHub REST base.
GITHUB_API_BASE = "https://api.github.com"
