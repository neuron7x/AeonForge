from celery import Celery
from datetime import datetime, timezone

from app.config import settings
from app.quality import run_auto_qc
from app.reputation import apply_reputation_event

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
    from app.models import Assignment, Batch, Payment, Task
    from app.github_verifier import verify_github_pr
    from app.monitoring import log, assignments_approved

    async def _run():
        async with AsyncSessionLocal() as session:
            res = await session.execute(select(Assignment).filter_by(id=assignment_id))
            a = res.scalar_one_or_none()
            if not a:
                return

            await session.refresh(a)
            await session.refresh(a, attribute_names=["user", "task"])

            verification_passed = True
            verification_reason = "text"
            artifact_text = a.submission_payload or ""

            if a.evidence_url:
                result = await verify_github_pr(a.evidence_url)
                verification_passed = result.get("verified", False)
                verification_reason = result.get("message", "github")
                if result.get("artifact_text"):
                    artifact_text = artifact_text or result["artifact_text"]
                a.verified_by = "auto-github"
            else:
                a.verified_by = "direct-submit"

            if not artifact_text.strip():
                artifact_text = ""
                verification_passed = False
                verification_reason = "empty_submission"

            qc_report = run_auto_qc(artifact_text)
            if not artifact_text:
                qc_report.setdefault("passed", False)
                qc_report.setdefault("reasons", []).append("empty_submission")

            a.qc_report = qc_report
            a.auto_qc_passed = qc_report.get("passed", False)
            a.reviewed_at = datetime.now(timezone.utc)

            late_submission = bool(a.submitted_at and a.due_at and a.submitted_at > a.due_at)

            if verification_passed and qc_report.get("passed", False):
                a.status = "approved"
                await session.commit()
                assignments_approved.inc()
                log.info("assignment_approved", assignment_id=assignment_id)
            else:
                a.status = "rejected"
                await session.commit()
                log.info(
                    "assignment_rejected",
                    assignment_id=assignment_id,
                    reason=verification_reason,
                    qc_reasons=qc_report.get("reasons", []),
                )

            await apply_reputation_event(
                session,
                a.user,
                accepted=a.status == "approved",
                rejected=a.status == "rejected",
                quality_bonus=qc_report.get("quality_bonus", False),
                late=late_submission,
            )
            await session.commit()

            if a.status != "approved" or not a.batch_id:
                return

            res2 = await session.execute(
                select(func.count()).select_from(Assignment).filter_by(batch_id=a.batch_id)
            )
            total = res2.scalar() or 0
            res3 = await session.execute(
                select(func.count()).select_from(Assignment).filter_by(batch_id=a.batch_id, status="approved")
            )
            done = res3.scalar() or 0
            if done != total or total == 0:
                return

            resb = await session.execute(select(Batch).filter_by(id=a.batch_id))
            batch = resb.scalar_one_or_none()
            if not batch or batch.status == "completed":
                return

            batch.status = "completed"
            await session.commit()

            res_assignments = await session.execute(
                select(Assignment, Task)
                .join(Task, Task.id == Assignment.task_id)
                .where(Assignment.batch_id == batch.id)
            )
            total_cents = 0
            for assignment_row, task_row in res_assignments.fetchall():
                reward = task_row.reward_cents
                if assignment_row.qc_report and assignment_row.qc_report.get("quality_bonus"):
                    reward += task_row.bonus_quality_cents
                if (
                    assignment_row.submitted_at
                    and assignment_row.due_at
                    and task_row.deadline_minutes
                    and (assignment_row.due_at - assignment_row.submitted_at).total_seconds()
                    >= 0.5 * task_row.deadline_minutes * 60
                ):
                    reward += task_row.bonus_speed_cents
                total_cents += reward

            p = Payment(user_id=a.user_id, batch_id=batch.id, amount_cents=total_cents, status="pending")
            session.add(p)
            await session.commit()
            trigger_payout.delay(p.id)

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
