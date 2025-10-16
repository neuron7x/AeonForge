import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("WEBHOOK_BASE_URL", "https://example.com")
os.environ.setdefault("WEBHOOK_SECRET", "secret")
os.environ.setdefault("GITHUB_TOKEN", "test")

from app.github_verifier import extract_pr_info, verify_github_pr


@pytest.mark.asyncio
async def test_extract_pr_info():
    url = "https://github.com/owner/repo/pull/123"
    result = await extract_pr_info(url)
    assert result == ("owner/repo", 123)

    invalid_url = "https://github.com/invalid"
    result = await extract_pr_info(invalid_url)
    assert result is None


@pytest.mark.asyncio
async def test_verify_github_pr_merged():
    with patch("app.github_verifier.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "merged": True,
            "user": {"login": "author"},
            "title": "Fix bug",
        }
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        result = await verify_github_pr("https://github.com/owner/repo/pull/123")
        assert result["verified"] is True
        assert result["merged"] is True


@pytest.mark.asyncio
async def test_verify_github_pr_not_merged():
    with patch("app.github_verifier.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "merged": False,
            "state": "open",
            "user": {"login": "author"},
            "title": "WIP",
        }
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        result = await verify_github_pr("https://github.com/owner/repo/pull/456")
        assert result["verified"] is False
        assert result["message"] == "PR not merged"
