import pytest
from app.github_verifier import extract_pr_info, verify_github_pr
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_extract_pr_info():
    url = "https://github.com/owner/repo/pull/123"
    assert extract_pr_info(url) == ("owner/repo", 123)
    assert extract_pr_info("https://github.com/invalid") is None

@pytest.mark.asyncio
async def test_verify_github_pr_merged():
    with patch("app.github_verifier.httpx.AsyncClient") as mock_client:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "merged": True,
            "user": {"login": "author"},
            "title": "Fix",
            "_links": {"self": {"href": "https://api.github.com/repos/owner/repo/pulls/123"}}
        }
        mock_files = AsyncMock()
        mock_files.status_code = 200
        mock_files.json.return_value = [{"filename": "a.txt"}]

        inst = AsyncMock()
        inst.get = AsyncMock(side_effect=[mock_resp, mock_files])
        mock_client.return_value.__aenter__.return_value = inst

        res = await verify_github_pr("https://github.com/owner/repo/pull/123")
        assert res["verified"] is True
        assert res["merged"] is True
