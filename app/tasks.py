from celery import Celery
from app.config import settings

celery_app = Celery(
    "promptops",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30*60,
    task_soft_time_limit=25*60,
)

celery_app.conf.task_routes = {
    "app.tasks.verify_assignment": {"queue": "default"},
    "app.tasks.trigger_payout": {"queue": "default"},
}

@celery_app.task(name="app.tasks.verify_assignment", bind=True, max_retries=3)
def verify_assignment(self, assignment_id: int):
    import asyncio
    from sqlalchemy import select, func
    from app.db import AsyncSessionLocal
    from app.models import Assignment, Batch, Payment
    from app.github_verifier import verify_github_pr
    from app.monitoring import log, assignments_approved

    async def _run():
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(Assignment).filter_by(id=assignment_id))
            a = res.scalar_one_or_none()
            if not a or not a.evidence_url:
                return
            result = await verify_github_pr(a.evidence_url)
            if result.get("verified"):
                a.status = "approved"
                a.verified_by = "auto-github"
                await session.commit()
                assignments_approved.inc()
                log.info("assignment_approved", assignment_id=assignment_id)

                # If batch exists, check completion
                if a.batch_id:
                    res2 = await session.execute(select(func.count()).select_from(Assignment).filter_by(batch_id=a.batch_id))
                    total = res2.scalar() or 0
                    res3 = await session.execute(select(func.count()).select_from(Assignment).filter_by(batch_id=a.batch_id, status="approved"))
                    done = res3.scalar() or 0
                    if done == total and total > 0:
                        # Mark batch completed and trigger payout once
                        resb = await session.execute(select(Batch).filter_by(id=a.batch_id))
                        batch = resb.scalar_one_or_none()
                        if batch and batch.status != "completed":
                            batch.status = "completed"
                            await session.commit()
                            # Sum rewards
                            from sqlalchemy import select
                            res4 = await session.execute(select(func.sum(Assignment.task.property.mapper.class_.reward_cents)).filter_by(batch_id=batch.id))
                            amount = res4.scalar() or 0
                            p = Payment(user_id=a.user_id, batch_id=batch.id, amount_cents=amount, status="pending")
                            session.add(p)
                            await session.commit()
                            trigger_payout.delay(p.id)
            else:
                a.status = "rejected"
                await session.commit()
                log.info("assignment_rejected", assignment_id=assignment_id, reason=result.get("message"))

    try:
        asyncio.run(_run())
    except Exception as exc:
        self.retry(exc=exc, countdown=60)

@celery_app.task(name="app.tasks.trigger_payout")
def trigger_payout(payment_id: int):
    import httpx, asyncio
    from sqlalchemy import select
    from app.db import AsyncSessionLocal
    from app.models import Payment
    from app.monitoring import log

    async def _run():
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(Payment).filter_by(id=payment_id))
            p = res.scalar_one_or_none()
            if not p:
                return
            if not settings.PAYOUT_WEBHOOK_URL:
                log.warning("payout_webhook_not_configured")
                p.status = "pending"
                await session.commit()
                return
            payload = {"user_id": p.user_id, "batch_id": p.batch_id, "amount_cents": p.amount_cents}
            headers = {"X-Payout-Secret": settings.PAYOUT_SECRET, "Content-Type": "application/json"}
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    r = await client.post(settings.PAYOUT_WEBHOOK_URL, json=payload, headers=headers)
                    if r.status_code in (200, 202):
                        p.status = "paid"
                    else:
                        p.status = "failed"
                        p.error_reason = f"HTTP {r.status_code}: {r.text[:500]}"
                    await session.commit()
            except Exception as e:
                p.status = "failed"
                p.error_reason = str(e)[:500]
                await session.commit()
                log.error("payout_error", error=str(e))

    asyncio.run(_run())
