import os, time, json
import http.client
import pytest

API_URL = os.getenv("API_URL")
HEALTH_PATH = os.getenv("HEALTH_PATH", "/health")

@pytest.mark.skipif(not API_URL, reason="API_URL is not provided")
def test_health_smoke():
    time.sleep(2)
    conn = http.client.HTTPConnection(API_URL.replace("http://","").replace("https://","").split('/')[0], timeout=5)
    conn.request("GET", HEALTH_PATH)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", errors="ignore")
    assert resp.status == 200, f"/health returned {resp.status}: {body[:200]}"
