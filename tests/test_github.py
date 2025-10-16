import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("WEBHOOK_BASE_URL", "https://example.com")
os.environ.setdefault("WEBHOOK_SECRET", "secret")
os.environ.setdefault("GITHUB_TOKEN", "test")

from app.github_verifier import extract_pr_info, verify_github_pr


@pytest.mark.asyncio
async def test_extract_pr_info_parses_url():
    result = await extract_pr_info("https://github.com/org/repo/pull/42")
    assert result == ("org/repo", 42)


@pytest.mark.asyncio
async def test_extract_pr_info_invalid_url():
    result = await extract_pr_info("https://github.com/org/repo")
    assert result is None


@pytest.mark.asyncio
async def test_verify_github_pr_handles_org_membership():
    with patch("app.github_verifier.httpx.AsyncClient") as mock_client:
        pull_response = MagicMock()
        pull_response.status_code = 200
        pull_response.json.return_value = {
            "merged": True,
            "user": {"login": "author"},
            "title": "Awesome feature",
        }

        org_response = MagicMock()
        org_response.status_code = 204

        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=[pull_response, org_response])
        mock_client.return_value.__aenter__.return_value = instance

        with patch("app.config.settings.GITHUB_ORG", "promptops"):
            result = await verify_github_pr("https://github.com/org/repo/pull/101")

        assert result["verified"] is True
        assert result["message"] == "PR merged and author is org member"
