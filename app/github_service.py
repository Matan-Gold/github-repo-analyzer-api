"""GitHub ingestion: URL parsing, tree fetch, filtering, and file content retrieval."""

from __future__ import annotations

import base64
import concurrent.futures
import os
import re
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote

import requests

from app import config
from app.models import AppError, RepoTreeItem, SelectedFile
from app.utils import is_deprioritized, should_skip_file


GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)

DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "Dockerfile",
    "docker-compose.yml",
}


class GitHubService:
    def __init__(self) -> None:
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "github-repo-analyzer-api",
        }
        github_token = os.getenv(config.GITHUB_TOKEN_ENV, "").strip()
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        self.session.headers.update(headers)

    def parse_github_url(self, github_url: str) -> tuple[str, str]:
        normalized = github_url.strip()
        if normalized.endswith("/"):
            normalized = normalized[:-1]
        match = GITHUB_URL_RE.match(normalized)
        if not match:
            raise AppError(
                code="INVALID_GITHUB_URL",
                message="Expected URL like https://github.com/{owner}/{repo}",
                status_code=400,
                details={"github_url": github_url},
            )
        return match.group("owner"), match.group("repo")

    def _request_json(self, url: str) -> dict[str, Any]:
        # Exponential backoff for rate-limit responses: 1s, 2s, 4s.
        timeout_attempted = False
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=config.TIMEOUT_GITHUB_SECONDS)
            except requests.Timeout as exc:
                if timeout_attempted:
                    raise AppError(
                        code="GITHUB_TIMEOUT",
                        message="GitHub API request timed out.",
                        status_code=504,
                        details={"url": url},
                    ) from exc
                timeout_attempted = True
                time.sleep(1)
                continue
            except requests.RequestException as exc:
                raise AppError(
                    code="GITHUB_API_ERROR",
                    message="GitHub API request failed.",
                    status_code=502,
                    details={"url": url, "error": str(exc)},
                ) from exc

            if response.status_code in (403, 429):
                remaining = response.headers.get("X-RateLimit-Remaining")
                if remaining == "0" or response.status_code == 429:
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    raise AppError(
                        code="GITHUB_RATE_LIMIT",
                        message="GitHub API rate limit exceeded.",
                        status_code=429,
                    )

            if response.status_code == 404:
                raise AppError(
                    code="REPO_NOT_FOUND",
                    message="Repository not found.",
                    status_code=404,
                )

            if response.status_code >= 400:
                raise AppError(
                    code="GITHUB_API_ERROR",
                    message="GitHub API returned an error.",
                    status_code=502,
                    details={"status_code": response.status_code, "url": url},
                )

            try:
                return response.json()
            except ValueError as exc:
                raise AppError(
                    code="GITHUB_API_ERROR",
                    message="GitHub API returned invalid JSON.",
                    status_code=502,
                    details={"url": url},
                ) from exc

        raise AppError(code="GITHUB_API_ERROR", message="GitHub API request failed.", status_code=502)

    def get_repo_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}"
        return self._request_json(url)

    def get_repo_tree(self, owner: str, repo: str, ref: str) -> list[RepoTreeItem]:
        url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{quote(ref)}?recursive=1"
        data = self._request_json(url)
        entries = data.get("tree", [])
        files: list[RepoTreeItem] = []
        for entry in entries:
            if entry.get("type") != "blob":
                continue
            path = str(entry.get("path", ""))
            if not path:
                continue
            size = int(entry.get("size", 0) or 0)
            files.append(RepoTreeItem(path=path, size=size, type="blob"))
        if not files:
            raise AppError(
                code="GITHUB_API_ERROR",
                message="Repository appears empty or has no files.",
                status_code=422,
            )
        return files

    def prefilter_tree(self, files: Iterable[RepoTreeItem]) -> list[RepoTreeItem]:
        # Keep only potentially useful source/config files.
        filtered: list[RepoTreeItem] = []
        for item in files:
            if should_skip_file(item.path):
                continue
            filtered.append(item)
        return filtered

    def _fetch_single_content(self, owner: str, repo: str, ref: str, item: RepoTreeItem) -> SelectedFile:
        # Hard byte limit: oversized files are skipped rather than fetched.
        if item.size > config.MAX_FILE_BYTES:
            return SelectedFile(path=item.path, skipped=True, skip_reason="FILE_TOO_LARGE")

        url = f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{quote(item.path)}?ref={quote(ref)}"
        timeout_attempted = False
        for _attempt in range(2):
            try:
                response = self.session.get(url, timeout=config.TIMEOUT_GITHUB_SECONDS)
            except requests.Timeout:
                if timeout_attempted:
                    return SelectedFile(path=item.path, skipped=True, skip_reason="GITHUB_TIMEOUT")
                timeout_attempted = True
                time.sleep(1)
                continue
            except requests.RequestException:
                return SelectedFile(path=item.path, skipped=True, skip_reason="GITHUB_API_ERROR")

            if response.status_code == 404:
                return SelectedFile(path=item.path, skipped=True, skip_reason="MISSING_FILE")
            if response.status_code >= 400:
                return SelectedFile(path=item.path, skipped=True, skip_reason=f"HTTP_{response.status_code}")

            try:
                data = response.json()
            except ValueError:
                return SelectedFile(path=item.path, skipped=True, skip_reason="INVALID_JSON")

            encoding = data.get("encoding")
            if encoding == "base64":
                encoded = data.get("content", "")
                try:
                    # Graceful decoding keeps pipeline resilient on mixed encodings.
                    decoded = base64.b64decode(encoded, validate=False)
                    return SelectedFile(path=item.path, content=decoded.decode("utf-8", errors="replace"))
                except Exception:
                    return SelectedFile(path=item.path, skipped=True, skip_reason="FILE_DECODE_ERROR")

            content = data.get("content")
            if isinstance(content, str):
                return SelectedFile(path=item.path, content=content)

            return SelectedFile(path=item.path, skipped=True, skip_reason="UNSUPPORTED_CONTENT_ENCODING")

        return SelectedFile(path=item.path, skipped=True, skip_reason="GITHUB_TIMEOUT")

    def fetch_selected_files(
        self,
        *,
        owner: str,
        repo: str,
        ref: str,
        selected_paths: list[str],
        tree_map: dict[str, RepoTreeItem],
    ) -> list[SelectedFile]:
        selected_items = [tree_map[path] for path in selected_paths if path in tree_map]
        if not selected_items:
            return []

        # Concurrent fetch, then reorder to planner-selected path order.
        results: list[SelectedFile] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.CONCURRENCY_GITHUB) as executor:
            futures = [
                executor.submit(self._fetch_single_content, owner, repo, ref, item)
                for item in selected_items
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        order = {path: idx for idx, path in enumerate(selected_paths)}
        results.sort(key=lambda item: order.get(item.path, 10_000))
        return results

    def fallback_file_selection(self, files: list[RepoTreeItem], limit: int = config.MAX_SELECTED_FILES) -> list[str]:
        if not files:
            return []

        paths = [item.path for item in files]
        selected: list[str] = []
        folder_counts: dict[str, int] = {}

        def _can_take(path: str) -> bool:
            folder = os.path.dirname(path)
            return folder_counts.get(folder, 0) < 3

        def _add(path: str) -> None:
            if path in selected:
                return
            if not _can_take(path):
                return
            selected.append(path)
            folder = os.path.dirname(path)
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        # A) README/docs
        for path in paths:
            lower = path.lower()
            if lower == "readme.md" or lower.startswith("docs/") and (
                lower.endswith("index.md") or lower.endswith("readme.md")
            ):
                _add(path)
            if len(selected) >= limit:
                return selected[:limit]

        # B) Dependencies/build
        for path in paths:
            if os.path.basename(path) in DEPENDENCY_FILES:
                _add(path)
            if len(selected) >= limit:
                return selected[:limit]

        # C) Entrypoints/wiring
        for path in paths:
            base = os.path.basename(path).lower()
            lower = path.lower()
            if base.startswith(("main.", "app.", "server.", "index.")):
                _add(path)
            elif lower.startswith("src/main.") or lower.endswith("/main.go"):
                _add(path)
            elif "wsgi" in base or "asgi" in base:
                _add(path)
            if len(selected) >= limit:
                return selected[:limit]

        # D) Core shallow source
        for path in paths:
            if path in selected or is_deprioritized(path):
                continue
            lower = path.lower()
            depth = lower.count("/")
            if (lower.startswith("src/") or lower.startswith("app/") or lower.startswith("lib/")) and depth <= 3:
                _add(path)
            if len(selected) >= limit:
                return selected[:limit]

        # Fill remaining slots with the smallest non-deprioritized files.
        remaining = sorted(files, key=lambda item: item.size)
        for item in remaining:
            if item.path in selected or is_deprioritized(item.path):
                continue
            _add(item.path)
            if len(selected) >= limit:
                break

        return selected[:limit]
