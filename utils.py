"""Utility helpers for token budgeting, chunking, and path heuristics."""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from typing import Optional

import config


SKIP_DIR_PREFIXES = (
    "node_modules/",
    "vendor/",
    "third_party/",
    "dist/",
    "build/",
    "out/",
    "target/",
    "coverage/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".git/",
)

SKIP_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".parquet",
}

DEPRIORITIZED_LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        return len(encoding.encode(text))
    except Exception:
        # Fallback approximation when tokenizer package/model mapping is unavailable.
        return max(1, len(text) // 4)


def chunk_text_by_lines(text: str, target_tokens: int = config.CHUNK_TOKENS) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > target_tokens:
            chunks.append("".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        chunks.append("".join(current))
    return chunks


def should_skip_file(path: str) -> bool:
    norm = path.strip("/")
    for prefix in SKIP_DIR_PREFIXES:
        marker = prefix.strip("/")
        # Match both top-level and nested generated/cache/vendor directories.
        if norm.startswith(prefix) or f"/{marker}/" in f"/{norm}/":
            return True

    lower_name = norm.lower()
    if lower_name.endswith(".min.js") or lower_name.endswith(".map"):
        return True

    ext = PurePosixPath(norm).suffix.lower()
    return ext in SKIP_EXTENSIONS


def is_deprioritized(path: str) -> bool:
    norm = path.strip("/")
    base = os.path.basename(norm)
    lower = norm.lower()
    if base in DEPRIORITIZED_LOCKFILES:
        return True
    return (
        lower.startswith("tests/")
        or "test" in base.lower()
        or "spec" in base.lower()
    )


def summarize_for_prompt(text: str, max_chars: int = 10_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def truncate_words(text: str, max_words: int = 120) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" .,;") + "."


def clamp_list(items: list[str], minimum: int, maximum: int) -> list[str]:
    cleaned = [item.strip() for item in items if item and item.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in cleaned:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if len(deduped) > maximum:
        return deduped[:maximum]
    return deduped


def infer_languages(paths: list[str]) -> list[str]:
    ext_to_lang = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".jsx": "JavaScript",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".kt": "Kotlin",
        ".rb": "Ruby",
        ".php": "PHP",
        ".cs": "C#",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++",
        ".sh": "Shell",
        ".yml": "YAML",
        ".yaml": "YAML",
    }
    langs: list[str] = []
    seen: set[str] = set()
    for path in paths:
        ext = PurePosixPath(path).suffix.lower()
        lang = ext_to_lang.get(ext)
        if lang and lang not in seen:
            seen.add(lang)
            langs.append(lang)
    return langs


def normalize_tech_name(value: str) -> str:
    cleaned = re.sub(r"[<>=~!].*$", "", value).strip()
    cleaned = cleaned.strip("\"' ")
    return cleaned
