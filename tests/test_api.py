import json
import os
from datetime import datetime

TEST_JWT_KEYSET = {
    "current": "kid-primary",
    "keys": [
        {
            "kid": "kid-primary",
            "secret": "primary-secret",
            "activated_at": "2024-01-01T00:00:00Z",
        },
        {
            "kid": "kid-legacy",
            "secret": "legacy-secret",
            "activated_at": "2023-10-01T00:00:00Z",
        },
    ],
}

os.environ.setdefault("JWT_SECRETS_BACKEND", "env")
os.environ["JWT_KEYSET_JSON"] = json.dumps(TEST_JWT_KEYSET)
os.environ.setdefault("JWT_ALG", "HS256")

from fastapi.testclient import TestClient
from jose import jwt

from infrastructure.api import app


def token(secret="primary-secret", kid="kid-primary"):
    return jwt.encode({"sub": "test"}, secret, algorithm="HS256", headers={"kid": kid})

def test_health_and_metrics():
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        headers = {"Authorization": f"Bearer {token()}"}
        assert c.get("/system/status", headers=headers).status_code == 200

        metrics_response = c.get("/metrics")
        assert metrics_response.status_code == 200
        body = metrics_response.text
        assert "cbc_requests_total" in body
        assert "cbc_jwt_keys_loaded" in body
        assert "cbc_jwt_active_key_age_seconds" in body
        assert "cbc_jwt_auth_failures_total" in body

def test_biometric_and_delegation_flow():
    with TestClient(app) as c:
        headers = {"Authorization": f"Bearer {token()}"}
        payload = {
            "user_id": "u1",
            "hrv_sdnn": 52.0,
            "hrv_rmssd": 30.0,
            "rhr": 72.0,
            "sleep_duration": 7.0,
            "sleep_efficiency": 0.85,
            "waso": 35.0,
            "context_switches": 6,
            "timestamp": datetime.now().isoformat()
        }
        r = c.post("/biometric/submit", json=payload, headers=headers)
        assert r.status_code == 200, r.text
        eoi = r.json()["eoi_components"]["eoi"]
        assert 0 <= eoi <= 3

        r2 = c.post("/delegate/plan", json={
            "user_id":"u1",
            "task_type":"code_generation",
            "task_description":"Implement sorting algo",
            "observed_hrv": 70.0,
            "self_reported_load":"medium",
            "completion_time": 1.1
        }, headers=headers)
        assert r2.status_code == 200, r2.text
        js = r2.json()
        assert js["task_type"] == "code_generation"
        assert 0 <= js["recommended_ai_autonomy"] <= 1
