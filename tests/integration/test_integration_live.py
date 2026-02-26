import os

import httpx
import pytest


@pytest.mark.integration
def test_live_summarize_requests_repo():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("RUN_INTEGRATION is not enabled")
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY is not set")

    base_url = os.getenv("INTEGRATION_BASE_URL", "http://localhost:8000")
    response = httpx.post(
        f"{base_url}/summarize",
        json={"github_url": "https://github.com/psf/requests"},
        timeout=120.0,
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["summary"]) > 50
    assert len(payload["technologies"]) > 0
