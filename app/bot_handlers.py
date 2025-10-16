from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy import select
from telegram import Bot
from telegram.error import TelegramError

from app.config import settings
from app.db import AsyncSessionLocal
from app.models import Assignment, AuditLog, Task, User
from app.monitoring import assignments_submitted, log
from app.tasks import verify_assignment

bot = Bot(token=settings.BOT_TOKEN)


async def get_or_create_user(session, tg_user: dict) -> User:
    """Retrieve an existing user or create a new record."""
    result = await session.execute(select(User).filter_by(tg_id=tg_user["id"]))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            tg_id=tg_user["id"],
            username=tg_user.get("username", ""),
            display_name=(tg_user.get("first_name", "") + " " + tg_user.get("last_name", "")).strip(),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        log.info("user_created", tg_id=user.tg_id)

    return user


async def log_audit(session, user_id: int, action: str, description: str = "", metadata: str = "") -> None:
    """Persist audit events for key bot actions."""
    audit = AuditLog(
        user_id=user_id,
        action=action,
        description=description,
        metadata=metadata,
    )
    session.add(audit)
    await session.commit()


async def cmd_start(chat_id: int, tg_user: dict) -> None:
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(session, tg_user)
        await log_audit(session, user.id, "start", "User started bot")

    text = (
        "Привіт! 👋\n\n"
        "Я PromptOpsBot. Тут можна:\n"
        "• /tasks — переглянути доступні завдання\n"
        "• /take N — взяти N завдань\n"
        "• /my — мої активні завдання\n"
        "• /submit ID URL — надіслати доказ\n"
        "• /help — підказка"
    )

    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError as exc:  # pragma: no cover - Telegram failure path
        log.error("telegram_error", error=str(exc), chat_id=chat_id)


async def cmd_tasks(chat_id: int) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Task).filter_by(status="available").limit(20))
        tasks: List[Task] = result.scalars().all()

    if not tasks:
        text = "Наразі немає доступних завдань 😕"
    else:
        lines = [
            f"#{task.id} — {task.title}\n💰 {task.reward_cents // 100}$ | ⏰ {task.requirement}" for task in tasks
        ]
        text = "\n\n".join(lines)

    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError as exc:  # pragma: no cover - Telegram failure path
        log.error("telegram_error", error=str(exc), chat_id=chat_id)


async def cmd_take(chat_id: int, tg_user: dict, count: int = 1) -> None:
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(session, tg_user)

        result = await session.execute(select(Assignment).filter_by(user_id=user.id, status="assigned"))
        active_assignments = result.scalars().all()
        if len(active_assignments) >= settings.MAX_ASSIGN_PER_USER:
            await bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Ти маєш максимум активних завдань ({settings.MAX_ASSIGN_PER_USER})",
            )
            return

        result = await session.execute(select(Task).filter_by(status="available").limit(count))
        available_tasks = result.scalars().all()
        if not available_tasks:
            await bot.send_message(chat_id=chat_id, text="На жаль, завдань немає 😕")
            return

        due_at = datetime.now(timezone.utc) + timedelta(hours=settings.TASK_DEADLINE_HOURS)
        assignments = []
        for task in available_tasks:
            task.status = "assigned"
            assignment = Assignment(
                user_id=user.id,
                task_id=task.id,
                due_at=due_at,
                status="assigned",
            )
            session.add(assignment)
            assignments.append(assignment)

        await session.flush()
        assigned_ids = [assignment.id for assignment in assignments]
        await session.commit()
        await log_audit(session, user.id, "take_tasks", f"Took {len(assignments)} tasks", str(assigned_ids))

    text = (
        f"✅ Ти взяв {len(assigned_ids)} завдань!\n\n"
        f"⏰ Дедлайн: {due_at.strftime('%Y-%m-%d %H:%M')} UTC\n\n"
        f"Як надіслати доказ:\n"
        f"`/submit <assignment_id> <github_pr_url>`\n\n"
        f"Приклад:\n`/submit {assigned_ids[0]} https://github.com/owner/repo/pull/123`"
    )

    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except TelegramError as exc:  # pragma: no cover - Telegram failure path
        log.error("telegram_error", error=str(exc), chat_id=chat_id)


async def cmd_my(chat_id: int, tg_user: dict) -> None:
    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(session, tg_user)
        result = await session.execute(
            select(Assignment).filter_by(user_id=user.id).order_by(Assignment.assigned_at.desc())
        )
        assignments = result.scalars().all()

    if not assignments:
        text = "У тебе немає завдань 📋"
    else:
        lines = []
        for assignment in assignments:
            status_icon = {
                "assigned": "⏳",
                "submitted": "📤",
                "approved": "✅",
                "rejected": "❌",
            }.get(assignment.status, "❓")
            lines.append(
                f"{status_icon} ID: {assignment.id} | Task: {assignment.task_id}\n"
                f"   До: {assignment.due_at.strftime('%Y-%m-%d %H:%M')} UTC | Статус: {assignment.status}"
            )
        text = "\n\n".join(lines)

    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError as exc:  # pragma: no cover - Telegram failure path
        log.error("telegram_error", error=str(exc), chat_id=chat_id)


async def cmd_submit(chat_id: int, tg_user: dict, parts: list[str]) -> None:
    if len(parts) < 3:
        await bot.send_message(
            chat_id=chat_id,
            text="Формат: `/submit <assignment_id> <github_pr_url>`",
            parse_mode="Markdown",
        )
        return

    try:
        assignment_id = int(parts[1])
        evidence_url = parts[2]
    except (ValueError, IndexError):
        await bot.send_message(chat_id=chat_id, text="❌ Невірний формат")
        return

    async with AsyncSessionLocal() as session:
        user = await get_or_create_user(session, tg_user)
        result = await session.execute(select(Assignment).filter_by(id=assignment_id, user_id=user.id))
        assignment = result.scalar_one_or_none()

        if not assignment:
            await bot.send_message(chat_id=chat_id, text="❌ Це завдання не твоє або не існує")
            return

        if assignment.status != "assigned":
            await bot.send_message(chat_id=chat_id, text=f"❌ Це завдання вже має статус: {assignment.status}")
            return

        assignment.evidence_url = evidence_url
        assignment.status = "submitted"
        assignment.submitted_at = datetime.now(timezone.utc)
        await session.commit()
        await log_audit(
            session,
            user.id,
            "submit_evidence",
            f"Submitted assignment {assignment_id}",
            evidence_url,
        )
        assignments_submitted.inc()

    text = (
        f"✅ Доказ отримано для завдання #{assignment_id}\n"
        f"Посилання: {evidence_url}\n\n"
        f"⏳ Чекаємо на автоматичну верифікацію..."
    )

    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError as exc:  # pragma: no cover - Telegram failure path
        log.error("telegram_error", error=str(exc), chat_id=chat_id)

    verify_assignment.delay(assignment_id)


async def cmd_help(chat_id: int) -> None:
    text = (
        "📖 *Довідка PromptOpsBot*\n\n"
        "*Команди:*\n"
        "`/start` — привіт\n"
        "`/tasks` — доступні завдання\n"
        "`/take N` — взяти N завдань (за замовч. 1)\n"
        "`/my` — мої завдання\n"
        "`/submit ID URL` — надіслати доказ\n"
        "`/help` — ця довідка\n\n"
        "*Як це працює:*\n"
        "1. /take — береш завдання\n"
        "2. Робиш PR у вказаному репо\n"
        "3. /submit ID URL — надсилаєш PR посилання\n"
        "4. Бот автоматично перевіряє PR\n"
        "5. Якщо все ОК — отримуєш платіж 💰"
    )

    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    except TelegramError as exc:  # pragma: no cover - Telegram failure path
        log.error("telegram_error", error=str(exc), chat_id=chat_id)


async def handle_update(update_json: dict) -> None:
    try:
        if "message" in update_json:
            message = update_json["message"]
            chat_id = message["chat"]["id"]
            text = message.get("text", "").strip()
            tg_user = message.get("from", {})

            if text.startswith("/start"):
                await cmd_start(chat_id, tg_user)
            elif text.startswith("/tasks"):
                await cmd_tasks(chat_id)
            elif text.startswith("/take"):
                parts = text.split()
                count = int(parts[1]) if len(parts) > 1 else 1
                count = min(count, settings.MAX_ASSIGN_PER_USER)
                await cmd_take(chat_id, tg_user, count)
            elif text.startswith("/my"):
                await cmd_my(chat_id, tg_user)
            elif text.startswith("/submit"):
                parts = text.split()
                await cmd_submit(chat_id, tg_user, parts)
            elif text.startswith("/help"):
                await cmd_help(chat_id)
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text="Я не розумію цю команду. /help для довідки",
                )
    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("handle_update_error", error=str(exc), update=str(update_json))
