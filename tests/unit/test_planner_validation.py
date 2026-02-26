import json

from fastapi.testclient import TestClient

import main
from llm_service import LLMService
from models import AppError, RepoTreeItem, SelectedFile


def test_llm_invalid_json_retries_once(monkeypatch):
    service = object.__new__(LLMService)
    service.model = "dummy"

    calls = {"count": 0}

    def fake_single_call(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "not json"
        return json.dumps({"important_files": ["README.md"]})

    monkeypatch.setattr(service, "_single_call", fake_single_call)
    result = service.call_json(system_prompt="s", user_prompt="u", max_output_tokens=50)

    assert calls["count"] == 2
    assert result == {"important_files": ["README.md"]}


def test_llm_invalid_json_failure_raises_structured_error(monkeypatch):
    service = object.__new__(LLMService)
    service.model = "dummy"

    def fake_single_call(**_kwargs):
        return "still-not-json"

    monkeypatch.setattr(service, "_single_call", fake_single_call)

    try:
        service.call_json(system_prompt="s", user_prompt="u", max_output_tokens=50)
        assert False, "Expected AppError"
    except AppError as err:
        assert err.code == "LLM_INVALID_RESPONSE"


class FakeGitHubService:
    def parse_github_url(self, _url):
        return "psf", "requests"

    def get_repo_metadata(self, _owner, _repo):
        return {"default_branch": "main"}

    def get_repo_tree(self, _owner, _repo, _ref):
        return [RepoTreeItem(path="README.md", size=10, type="blob")]

    def prefilter_tree(self, files):
        return list(files)

    def fetch_selected_files(self, **_kwargs):
        return [SelectedFile(path="README.md", content="# Repo")]

    def fallback_file_selection(self, _files, limit=10):
        return ["README.md"][:limit]


class MissingKeyLLMService:
    def __init__(self):
        self.calls = 0

    def call_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"important_files": ["README.md"]}
        return {"summary": "x", "technologies": ["Python"]}  # Missing "structure".


def test_missing_required_keys_returns_structured_error(monkeypatch):
    fake_llm = MissingKeyLLMService()
    monkeypatch.setattr("summarizer.GitHubService", FakeGitHubService)
    monkeypatch.setattr("summarizer.LLMService", lambda: fake_llm)

    client = TestClient(main.app)
    response = client.post("/summarize", json={"github_url": "https://github.com/psf/requests"})

    # Planner call + final call; final response remains invalid and must fail with structured error.
    assert fake_llm.calls == 2
    assert response.status_code == 502
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "LLM_INVALID_RESPONSE"
