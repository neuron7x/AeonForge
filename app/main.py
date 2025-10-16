from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import select, func

from app.config import settings
from app.db import init_db, close_db, AsyncSessionLocal
from app.bot_handlers import handle_update
from app.monitoring import log, webhook_requests, active_assignments
from app.models import Assignment

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info("application_started", environment=settings.ENVIRONMENT)
    yield
    await close_db()
    log.info("application_shutdown")

app = FastAPI(title="PromptOps Bot", version="1.0.0")
app.router.lifespan_context = lifespan

@app.post(settings.WEBHOOK_PATH)
async def webhook(request: Request):
    webhook_requests.labels(status="processing").inc()
    try:
        # Telegram sends a static secret token header — compare directly
        secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if settings.WEBHOOK_SECRET and secret_header != settings.WEBHOOK_SECRET:
            webhook_requests.labels(status="rejected").inc()
            raise HTTPException(status_code=403, detail="Invalid secret token")

        update_json = await request.json()
        await handle_update(update_json)

        webhook_requests.labels(status="ok").inc()
        return PlainTextResponse("ok")
    except Exception as e:
        log.error("webhook_error", error=str(e))
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
    except Exception as e:
        log.error("readiness_check_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Not ready")

@app.get("/metrics")
async def metrics():
    try:
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(func.count()).select_from(Assignment).filter_by(status="assigned"))
            active_assignments.set(res.scalar() or 0)
    except Exception as e:
        log.error("metrics_error", error=str(e))
    data = generate_latest()
    return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    return {
        "name": "PromptOps Bot",
        "endpoints": {"health": "/health", "ready": "/ready", "metrics": "/metrics", "webhook": settings.WEBHOOK_PATH}
    }
