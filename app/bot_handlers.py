import json
from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlalchemy import select, text, func
from telegram import Bot
from telegram.error import TelegramError

from app.db import AsyncSessionLocal
from app.models import User, Task, Assignment, Batch, AuditLog, Payment
from app.config import settings
from app.monitoring import log, assignments_submitted
from app.tasks import verify_assignment
from app.utils import rate_limit_take
from app.reputation import refresh_user_level
from app.task_spec import build_task_template

bot = Bot(token=settings.BOT_TOKEN)

async def get_or_create_user(session, tg_user: dict) -> User:
    res = await session.execute(select(User).filter_by(tg_id=tg_user["id"]))
    u = res.scalar_one_or_none()
    if not u:
        u = User(
            tg_id=tg_user["id"],
            username=tg_user.get("username", ""),
            display_name=(tg_user.get("first_name", "") + " " + tg_user.get("last_name", "")).strip(),
        )
        session.add(u)
        await session.commit()
        await session.refresh(u)
        log.info("user_created", tg_id=u.tg_id)
    return u

async def log_audit(session, user_id: int, action: str, description: str = "", metadata: str = ""):
    al = AuditLog(user_id=user_id, action=action, description=description, metadata=metadata)
    session.add(al)
    await session.commit()

async def send(chat_id: int, text: str, markdown=False):
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=("Markdown" if markdown else None))
    except TelegramError as e:
        log.error("telegram_error", error=str(e), chat_id=chat_id)

async def cmd_start(chat_id: int, tg_user: dict):
    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        await refresh_user_level(session, u)
        await log_audit(session, u.id, "start", "User started bot")
    await send(
        chat_id,
        "Привіт! Використовуй /task щоб отримати бриф, /tasks — перелік, /take N — взяти пачку, /my — прогрес, /stats — аналітика, /submit <ID> <дані> — здати.",
    )

async def cmd_tasks(chat_id: int, tg_user: dict):
    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        await refresh_user_level(session, u)
        res = await session.execute(
            select(Task)
            .filter_by(status="available")
            .where(Task.level_required <= u.level)
            .order_by(Task.created_at.asc())
            .limit(20)
        )
        tasks = res.scalars().all()
    if not tasks:
        await send(chat_id, "Наразі немає доступних завдань 😕")
    else:
        lines = [
            f"#{t.id} — {t.title}\n💰 {t.reward_cents/100:.2f}$ | 📌 {t.requirement} | lvl ≥ {t.level_required}"
            for t in tasks
        ]
        await send(chat_id, "\n\n".join(lines))


async def cmd_task(chat_id: int, tg_user: dict):
    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        await refresh_user_level(session, u)
        res = await session.execute(
            select(Task)
            .filter_by(status="available")
            .where(Task.level_required <= u.level)
            .order_by(Task.created_at.asc())
            .limit(1)
        )
        task = res.scalar_one_or_none()
    if not task:
        await send(chat_id, "Завдань у твоєму рівні поки немає. Повернись згодом!")
        return
    template = build_task_template(task)
    template_text = json.dumps(template, ensure_ascii=False, indent=2)
    await send(chat_id, f"🧾 Бриф:\n{template_text}")

async def cmd_take(chat_id: int, tg_user: dict, count: int = 1):
    if count <= 0: count = 1
    if count > settings.MAX_ASSIGN_PER_USER: count = settings.MAX_ASSIGN_PER_USER

    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        await refresh_user_level(session, u)

        # Rate limit
        if not await rate_limit_take(u.id):
            await send(chat_id, "⏱ Занадто часто. Спробуй пізніше.")
            return

        # Check current active count
        res = await session.execute(select(func.count()).select_from(Assignment).filter_by(user_id=u.id, status="assigned"))
        active_count = res.scalar() or 0
        if active_count >= settings.MAX_ASSIGN_PER_USER:
            await send(chat_id, f"⚠️ Ліміт активних завдань: {settings.MAX_ASSIGN_PER_USER}")
            return

        # Atomic assignment using SKIP LOCKED
        to_take = min(count, settings.MAX_ASSIGN_PER_USER - active_count)
        due_at = datetime.now(timezone.utc) + timedelta(hours=settings.TASK_DEADLINE_HOURS)

        async with session.begin():
            sql = text("""
                UPDATE tasks
                SET status='assigned'
                WHERE id IN (
                    SELECT id FROM tasks
                    WHERE status='available' AND level_required <= :level
                    FOR UPDATE SKIP LOCKED
                    LIMIT :n
                )
                RETURNING id
            """)
            res_ids = await session.execute(sql, {"n": to_take, "level": u.level})
            task_ids = [row[0] for row in res_ids.fetchall()]

            if not task_ids:
                await send(chat_id, "На жаль, завдань зараз немає.")
                return

            batch = Batch(user_id=u.id, due_at=due_at, total_tasks=len(task_ids), status="open")
            session.add(batch)
            await session.flush()

            for tid in task_ids:
                a = Assignment(user_id=u.id, task_id=tid, batch_id=batch.id, due_at=due_at, status="assigned")
                session.add(a)

        await session.commit()
        await log_audit(session, u.id, "take_tasks", f"Took {len(task_ids)}", str(task_ids))

        text_resp = (
            f"✅ Ти взяв {len(task_ids)} завдань у пачці #{batch.id}!\n\n"
            f"⏰ Дедлайн: {due_at.strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Надсилай роботи так: `/submit <assignment_id> <payload|url>`\n"
            f"Оплата надійде після виконання *всієї* пачки."
        )
        await send(chat_id, text_resp, markdown=True)

async def cmd_my(chat_id: int, tg_user: dict):
    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        res = await session.execute(select(Assignment).filter_by(user_id=u.id).order_by(Assignment.assigned_at.desc()))
        assigns = res.scalars().all()
    if not assigns:
        await send(chat_id, "У тебе немає завдань 📋")
        return
    lines = []
    for a in assigns:
        icon = {"assigned":"⏳","submitted":"📤","approved":"✅","rejected":"❌"}.get(a.status,"❓")
        lines.append(f"{icon} AID:{a.id} | Task:{a.task_id} | Batch:{a.batch_id or '-'} | До: {a.due_at.strftime('%Y-%m-%d %H:%M')} UTC | {a.status}")
    await send(chat_id, "\n\n".join(lines))

async def cmd_submit(chat_id: int, tg_user: dict, text: str):
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        await send(chat_id, "Формат: `/submit <assignment_id> <дані>`", markdown=True)
        return
    try:
        aid = int(parts[1])
        payload = parts[2].strip()
    except Exception:
        await send(chat_id, "❌ Невірний формат.")
        return

    if not payload:
        await send(chat_id, "❌ Немає даних для відправки.")
        return

    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        res = await session.execute(select(Assignment).filter_by(id=aid, user_id=u.id))
        a = res.scalar_one_or_none()
        if not a:
            await send(chat_id, "❌ Це завдання не знайдено або не твоє.")
            return
        if a.status != "assigned":
            await send(chat_id, f"❌ Стан завдання: {a.status}.")
            return
        if payload.lower().startswith("http"):
            a.evidence_url = payload
            a.submission_payload = None
        else:
            a.submission_payload = payload
            a.evidence_url = None
        a.status = "submitted"
        a.submitted_at = datetime.now(timezone.utc)
        await session.commit()
        assignments_submitted.inc()
        await log_audit(session, u.id, "submit_evidence", f"Assignment {aid}", payload)

    await send(chat_id, f"✅ Дані отримано для #{aid}. Іде перевірка...")
    verify_assignment.delay(aid)


async def _period_stats(session, user: User, start: datetime) -> Dict[str, float | int]:
    res = await session.execute(
        select(Assignment.status, Assignment.reviewed_at)
        .where(Assignment.user_id == user.id)
        .where(Assignment.reviewed_at.is_not(None))
        .where(Assignment.reviewed_at >= start)
    )
    statuses = [row[0] for row in res.fetchall()]
    approved = statuses.count("approved")
    rejected = statuses.count("rejected")

    payout_res = await session.execute(
        select(func.sum(Payment.amount_cents))
        .where(Payment.user_id == user.id)
        .where(Payment.status == "paid")
        .where(Payment.created_at >= start)
    )
    payout_cents = payout_res.scalar() or 0
    return {
        "approved": approved,
        "rejected": rejected,
        "payout_usd": round(payout_cents / 100, 2),
    }


async def cmd_stats(chat_id: int, tg_user: dict):
    now = datetime.now(timezone.utc)
    start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_week = now - timedelta(days=7)

    async with AsyncSessionLocal() as session:
        u = await get_or_create_user(session, tg_user)
        await refresh_user_level(session, u)
        today = await _period_stats(session, u, start_day)
        week = await _period_stats(session, u, start_week)
        level = u.level
        reputation = u.reputation_score

    text = (
        f"📊 *Статистика*\n"
        f"Рівень: {level} | Репутація: {reputation:.1f}\n\n"
        f"Сьогодні — ✅ {today['approved']} | ❌ {today['rejected']} | 💵 ${today['payout_usd']:.2f}\n"
        f"7 днів — ✅ {week['approved']} | ❌ {week['rejected']} | 💵 ${week['payout_usd']:.2f}"
    )
    await send(chat_id, text, markdown=True)

async def cmd_help(chat_id: int):
    text = (
        "📖 *PromptOpsBot — команди*\n\n"
        "`/task` — видати бриф у JSON\n"
        "`/tasks` — переглянути доступні задачі\n"
        "`/take N` — взяти N задач у пачку\n"
        "`/my` — всі мої задачі\n"
        "`/stats` — статистика та виплати\n"
        "`/submit ID payload` — надіслати артефакт або посилання\n\n"
        "Оплата — коли вся пачка завершена (approved)."
    )
    await send(chat_id, text, markdown=True)

async def handle_update(update_json: dict):
    try:
        if "message" in update_json:
            msg = update_json["message"]
            chat_id = msg["chat"]["id"]
            text = (msg.get("text") or "").strip()
            tg_user = msg.get("from", {})

            if text.startswith("/start"):
                await cmd_start(chat_id, tg_user)
            elif text.startswith("/task"):
                await cmd_task(chat_id, tg_user)
            elif text.startswith("/tasks"):
                await cmd_tasks(chat_id, tg_user)
            elif text.startswith("/take"):
                parts = text.split()
                n = int(parts[1]) if len(parts) > 1 else 1
                await cmd_take(chat_id, tg_user, n)
            elif text.startswith("/my"):
                await cmd_my(chat_id, tg_user)
            elif text.startswith("/stats"):
                await cmd_stats(chat_id, tg_user)
            elif text.startswith("/submit"):
                await cmd_submit(chat_id, tg_user, text)
            elif text.startswith("/help"):
                await cmd_help(chat_id)
            else:
                await send(chat_id, "Невідома команда. /help")
    except Exception as e:
        log.error("handle_update_error", error=str(e), update=str(update_json))
