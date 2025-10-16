import importlib
import os
import pytest

def _detect():
    candidates = os.environ.get("APP_MODULE_CANDIDATES",
        "api:app,infrastructure.api:app,app:app,main:app,server:app,src.api:app").split(",")
    for c in candidates:
        try:
            mod, app = c.split(":", 1)
            m = importlib.import_module(mod)
            a = getattr(m, app, None)
            if a is not None:
                return a
        except Exception:
            continue
    return None

@pytest.mark.skipif("fastapi" not in [m.__name__ for m in list(__import__('sys').modules.values()) if hasattr(m,'__name__')], reason="FastAPI not installed")
def test_health_endpoint_importable():
    try:
        from fastapi.testclient import TestClient  # type: ignore
    except Exception:
        pytest.skip("fastapi.testclient unavailable")
    app = _detect()
    if app is None:
        pytest.skip("No FastAPI app detected")
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
