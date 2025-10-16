from app.github_verifier import extract_pr_info

def test_extract_pr_info():
    assert extract_pr_info("https://github.com/owner/repo/pull/10") == ("owner/repo", 10)
    assert extract_pr_info("https://github.com/owner/repo") is None
