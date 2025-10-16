import asyncio

from celery import Celery

from app.config import settings

celery_app = Celery(
    "promptops",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
)

celery_app.conf.task_routes = {
    "app.tasks.verify_assignment": {"queue": "default"},
    "app.tasks.trigger_payout": {"queue": "default"},
}


@celery_app.task(name="app.tasks.verify_assignment", bind=True, max_retries=3)
def verify_assignment(self, assignment_id: int):
    """Verify GitHub PR evidence and update assignment status."""
    from sqlalchemy import select

    from app.db import AsyncSessionLocal
    from app.github_verifier import verify_github_pr
    from app.models import Assignment
    from app.monitoring import assignments_approved, log

    async def _verify() -> None:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Assignment).filter_by(id=assignment_id))
            assignment = result.scalar_one_or_none()

            if not assignment:
                log.warning("assignment_not_found", assignment_id=assignment_id)
                return

            if not assignment.evidence_url:
                log.warning("no_evidence_url", assignment_id=assignment_id)
                return

            verification = await verify_github_pr(assignment.evidence_url)

            if verification["verified"]:
                assignment.status = "approved"
                assignment.verified_by = "auto-github"
                await session.commit()
                log.info(
                    "assignment_approved",
                    assignment_id=assignment_id,
                    author=verification["author"],
                )
                assignments_approved.inc()
                trigger_payout.delay(assignment.user_id, assignment_id)
            else:
                assignment.status = "rejected"
                await session.commit()
                log.info(
                    "assignment_rejected",
                    assignment_id=assignment_id,
                    reason=verification["message"],
                )

    try:
        asyncio.run(_verify())
    except Exception as exc:  # pragma: no cover - Celery retry path
        from app.monitoring import log

        log.error("verify_assignment_error", assignment_id=assignment_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="app.tasks.trigger_payout")
def trigger_payout(user_id: int, assignment_id: int):
    """Send payout webhook."""
    import httpx

    from app.monitoring import log

    if not settings.PAYOUT_WEBHOOK_URL:
        log.warning("payout_webhook_not_configured")
        return

    payload = {
        "user_id": user_id,
        "assignment_id": assignment_id,
        "amount_cents": settings.PAYOUT_AMOUNT_CENTS,
    }

    headers = {
        "X-Payout-Secret": settings.PAYOUT_SECRET,
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            settings.PAYOUT_WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=15,
        )
        if response.status_code in (200, 202):
            log.info("payout_triggered", user_id=user_id, assignment_id=assignment_id)
        else:
            log.error("payout_failed", status=response.status_code, text=response.text)
    except Exception as exc:  # pragma: no cover - network failure logging
        log.error("payout_error", error=str(exc))
