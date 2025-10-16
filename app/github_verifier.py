import re
import time
from typing import Optional

# httpx is an optional dependency in the test environment.  We still want the
# module to remain importable – the tests monkeypatch ``httpx.AsyncClient`` –
# even when the real library is unavailable.  A light-weight shim keeps the
# public surface we rely on while nudging users towards installing httpx when
# they actually need to perform real network requests.
try:  # pragma: no cover - exercised implicitly via import
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - only triggered in slim envs
    class _MissingAsyncClient:
        """Minimal stand-in that fails fast when used without httpx installed."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple shim
            self._reason = "httpx is required to perform GitHub verification"

        async def __aenter__(self):  # pragma: no cover - defensive only
            raise RuntimeError(self._reason)

        async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
            return None

    class _HttpxFallback:
        AsyncClient = _MissingAsyncClient
        TimeoutException = TimeoutError

    httpx = _HttpxFallback()  # type: ignore

from app.config import settings
from app.monitoring import log, verification_duration


async def extract_pr_info(url: str) -> Optional[tuple[str, int]]:
    """Extract owner/repo and PR number from GitHub URL."""
    match = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url)
    if match:
        return match.group(1), int(match.group(2))
    return None


async def verify_github_pr(pr_url: str) -> dict:
    """Verify GitHub PR: check if merged and optionally organisation membership."""
    start = time.time()
    result: dict = {
        "verified": False,
        "merged": False,
        "author": "",
        "title": "",
        "message": "",
    }

    try:
        pr_info = await extract_pr_info(pr_url)
        if not pr_info:
            result["message"] = "Invalid GitHub URL format"
            return result

        owner_repo, pr_number = pr_info
        headers = {
            "Authorization": f"token {settings.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=15) as client:
            url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}"
            response = await client.get(url, headers=headers)

            if response.status_code == 404:
                result["message"] = "PR not found"
                return result

            if response.status_code != 200:
                result["message"] = f"GitHub API error: {response.status_code}"
                log.error("github_api_error", status=response.status_code, text=response.text)
                return result

            pr_payload = response.json()
            result["author"] = pr_payload.get("user", {}).get("login", "unknown")
            result["title"] = pr_payload.get("title", "")
            result["merged"] = pr_payload.get("merged", False)

            if settings.GITHUB_ORG:
                author_login = result["author"]
                org_check_url = f"https://api.github.com/orgs/{settings.GITHUB_ORG}/members/{author_login}"
                org_resp = await client.get(org_check_url, headers=headers)
                if org_resp.status_code == 204:
                    result["verified"] = result["merged"]
                    result["message"] = "PR merged and author is org member" if result["verified"] else "PR not merged yet"
                else:
                    result["message"] = f"Author {author_login} not member of {settings.GITHUB_ORG}"
            else:
                result["verified"] = result["merged"]
                result["message"] = "PR merged" if result["verified"] else "PR not merged"

    except httpx.TimeoutException:
        result["message"] = "GitHub API timeout"
        log.error("github_timeout")
    except Exception as exc:  # pragma: no cover - defensive logging
        result["message"] = f"Verification error: {exc}"
        log.error("github_verification_error", error=str(exc))
    finally:
        verification_duration.observe(time.time() - start)

    return result
