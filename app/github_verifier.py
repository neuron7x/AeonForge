import re
import httpx
import time
from app.config import settings
from app.monitoring import log, verification_duration

def extract_pr_info(url: str):
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url)
    if not m:
        return None
    owner_repo, prnum = m.group(1), int(m.group(2))
    return owner_repo, prnum

async def verify_github_pr(pr_url: str) -> dict:
    start = time.time()
    result = {
        "verified": False,
        "merged": False,
        "author": "",
        "title": "",
        "message": "",
        "artifact_text": "",
        "changed_files": [],
    }
    try:
        info = extract_pr_info(pr_url)
        if not info:
            result["message"] = "Invalid GitHub URL format"
            return result
        owner_repo, prnum = info
        headers = {
            "Authorization": f"token {settings.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            pr_url_api = f"https://api.github.com/repos/{owner_repo}/pulls/{prnum}"
            r = await client.get(pr_url_api, headers=headers)
            if r.status_code != 200:
                result["message"] = f"GitHub API error: {r.status_code}"
                log.error("github_api_error", status=r.status_code, text=r.text)
                return result
            pr = r.json()
            result["author"] = pr.get("user", {}).get("login", "unknown")
            result["title"] = pr.get("title", "")
            result["merged"] = pr.get("merged", False)

            # Additional checks: changed files > 0
            files_url = pr.get("_links", {}).get("self", {}).get("href") or pr_url_api
            files_resp = await client.get(f"{files_url}/files", headers=headers)
            changed_ok = False
            if files_resp.status_code == 200:
                files = files_resp.json()
                changed_ok = len(files) > 0
                patches = []
                filenames = []
                for file_entry in files:
                    filename = file_entry.get("filename")
                    if filename:
                        filenames.append(filename)
                    patch = file_entry.get("patch")
                    if patch:
                        patches.append(patch)
                if patches:
                    result["artifact_text"] = "\n".join(patches)[:50000]
                result["changed_files"] = filenames

            # Optional org membership check
            if settings.GITHUB_ORG:
                author = result["author"]
                org_resp = await client.get(f"https://api.github.com/orgs/{settings.GITHUB_ORG}/members/{author}", headers=headers)
                is_member = org_resp.status_code == 204
            else:
                is_member = True

            result["verified"] = bool(result["merged"] and changed_ok and is_member)
            if not changed_ok:
                result["message"] = "No changed files"
            elif not result["merged"]:
                result["message"] = "PR not merged"
            elif not is_member:
                result["message"] = f"Author not member of {settings.GITHUB_ORG}"
            else:
                result["message"] = "PR verified"

    except httpx.TimeoutException:
        result["message"] = "GitHub API timeout"
        log.error("github_timeout")
    except Exception as e:
        result["message"] = f"Verification error: {str(e)}"
        log.error("github_verification_error", error=str(e))
    finally:
        verification_duration.observe(time.time() - start)
    return result
