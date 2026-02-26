from fastapi.testclient import TestClient

import main
from models import RepoTreeItem, SelectedFile


class FakeGitHubService:
    def parse_github_url(self, _url):
        return "psf", "requests"

    def get_repo_metadata(self, _owner, _repo):
        return {"default_branch": "main"}

    def get_repo_tree(self, _owner, _repo, _ref):
        return [
            RepoTreeItem(path="README.md", size=100, type="blob"),
            RepoTreeItem(path="requirements.txt", size=100, type="blob"),
            RepoTreeItem(path="src/app.py", size=100, type="blob"),
        ]

    def prefilter_tree(self, files):
        return list(files)

    def fetch_selected_files(self, **_kwargs):
        return [
            SelectedFile(path="README.md", content="# Requests\nLibrary"),
            SelectedFile(path="requirements.txt", content="requests\nurllib3"),
            SelectedFile(path="src/app.py", content="def run():\n    return True"),
        ]

    def fallback_file_selection(self, _files, limit=10):
        return ["README.md", "requirements.txt", "src/app.py"][:limit]


class FakeLLMService:
    def call_json(self, **kwargs):
        prompt = kwargs.get("user_prompt", "")
        if "important_files" in prompt or "Repository file tree" in prompt:
            return {"important_files": ["README.md", "requirements.txt", "src/app.py"]}
        return {
            "summary": "Requests is a Python library for HTTP requests with a simple API.",
            "technologies": ["Python", "requests", "urllib3", "certifi", "api"],
            "structure": [
                "README.md: project overview.",
                "requirements.txt: dependency configuration.",
                "src/app.py: core source module.",
                "docs/: project docs.",
                "tests/: test suite.",
            ],
        }


def test_post_summarize_contract(monkeypatch):
    monkeypatch.setattr("summarizer.GitHubService", FakeGitHubService)
    monkeypatch.setattr("summarizer.LLMService", FakeLLMService)

    client = TestClient(main.app)
    response = client.post("/summarize", json={"github_url": "https://github.com/psf/requests"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"summary", "technologies", "structure"}
    assert isinstance(payload["summary"], str)
    assert isinstance(payload["technologies"], list)
    assert all(isinstance(item, str) for item in payload["technologies"])
    assert isinstance(payload["structure"], list)
    assert all(isinstance(item, str) for item in payload["structure"])
