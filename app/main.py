import hashlib
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select

from app.bot_handlers import handle_update
from app.config import settings
from app.db import AsyncSessionLocal, close_db, init_db
from app.models import Assignment
from app.monitoring import active_assignments, log, webhook_requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info("application_started", environment=settings.ENVIRONMENT)
    try:
        yield
    finally:
        await close_db()
        log.info("application_shutdown")


app = FastAPI(
    title="PromptOps Bot",
    version="1.0.0",
    description="Telegram bot для управління завданнями та PR верифікацією",
)
app.router.lifespan_context = lifespan


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    expected = hmac.new(settings.WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhook")
async def webhook(request: Request):
    webhook_requests.labels(status="processing").inc()

    try:
        signature = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        body = await request.body()

        if signature and not verify_webhook_signature(body, signature):
            webhook_requests.labels(status="rejected").inc()
            raise HTTPException(status_code=403, detail="Invalid signature")

        update_json = await request.json()
        await handle_update(update_json)

        webhook_requests.labels(status="ok").inc()
        return PlainTextResponse("ok")
    except Exception as exc:  # pragma: no cover - safety logging
        log.error("webhook_error", error=str(exc))
        webhook_requests.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Internal error")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "promptops-bot"}


@app.get("/ready")
async def ready():
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(select(func.count()).select_from(Assignment))
        return {"ready": True}
    except Exception as exc:  # pragma: no cover - readiness logging
        log.error("readiness_check_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Not ready")


@app.get("/metrics")
async def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.count()).select_from(Assignment).filter_by(status="assigned")
            )
            count = result.scalar() or 0
            active_assignments.set(count)
    except Exception as exc:  # pragma: no cover - metric logging
        log.error("metrics_error", error=str(exc))

    data = generate_latest()
    return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "name": "PromptOps Bot",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "webhook": "/webhook (POST)",
        },
    }
