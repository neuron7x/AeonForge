# PromptOps Bot 🤖

Production-ready Telegram bot для управління завданнями, верифікації GitHub PR та автоматичних платежів.

## ✨ Функціонал

- ✅ Telegram webhook інтеграція
- ✅ Управління завданнями (взяття, сабміт доказів)
- ✅ Автоматична верифікація GitHub PR
- ✅ Background задачі (Celery + Redis)
- ✅ Postgres БД з міграціями (Alembic)
- ✅ Prometheus metrics
- ✅ Структуроване логування
- ✅ CI/CD (GitHub Actions)
- ✅ Docker & docker-compose

## 🚀 Швидкий старт

### Передумови
- Docker & Docker Compose
- Python 3.11+
- Telegram Bot Token (від @BotFather)
- GitHub Token (для верифікації PR)

### Встановлення

```bash
# 1. Клонуємо
git clone https://github.com/your/prompt-ops-bot.git
cd prompt-ops-bot

# 2. Копіюємо конфіг
cp .env.template .env

# 3. Редагуємо .env (BOT_TOKEN, WEBHOOK_BASE_URL тощо)
nano .env

# 4. Встановлюємо
chmod +x install.sh
./install.sh
```

### Тестування

```bash
# 1. Перевір сервіси
docker-compose ps

# 2. Логи
docker-compose logs -f web

# 3. Пошли боту /start
```

## 📋 API Endpoints

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/webhook` | POST | Telegram webhook |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

## 🔧 Команди

```bash
# Логи
docker-compose logs -f web         # Вебсервер
docker-compose logs -f worker      # Celery воркер
docker-compose logs -f beat        # Celery beat

# БД
docker-compose exec web alembic upgrade head   # Міграції
docker-compose exec web alembic downgrade -1   # Откат

# Shell
docker-compose exec web python     # Python shell
docker-compose exec db psql -U bot bobdb      # Postgres shell

# Зупинка
docker-compose down -v             # Зупинити + видалити томи
```

## 📊 Структура Бази

### Users
```sql
id | tg_id | username | display_name | joined_at
```

### Tasks
```sql
id | title | text | requirement | status | reward_cents | created_at
```

### Assignments
```sql
id | user_id | task_id | status | evidence_url | submitted_at | verified_by
```

### Payments
```sql
id | user_id | assignment_id | amount_cents | status | payout_id
```

## 🤖 Telegram Commands

| Command | Опис |
|---------|------|
| `/start` | Привіт |
| `/tasks` | Список доступних завдань |
| `/take N` | Взяти N завдань |
| `/my` | Мої активні завдання |
| `/submit ID URL` | Надіслати доказ (PR) |
| `/help` | Довідка |

## 🔐 Безпека

- ✅ Webhook signature verification (HMAC-SHA256)
- ✅ Environment variables для секретів
- ✅ Admin-only операції
- ✅ Rate limiting
- ✅ Audit logging

## 📈 Моніторинг

### Prometheus metrics на `/metrics`
```
- webhook_requests_total
- assignments_submitted_total
- assignments_approved_total
- verification_duration_seconds
- active_assignments
```

### Структуроване логування (JSON)
```json
{"timestamp": "2024-01-15T10:30:00", "event": "assignment_approved", "user_id": 123}
```

## 🚢 Deployment

### Варіант 1: VPS з Docker

```bash
# На сервері
git clone ...
cd prompt-ops-bot
cp .env.template .env
# редагуємо .env
./install.sh
```

### Варіант 2: CI/CD автодеплой (GitHub Actions)

Налаштуй GitHub Secrets і CI буде автоматично деплоїти на push до main.

## 🧪 Тестування

```bash
# Unit tests
pytest tests/

# З покриттям
pytest tests/ --cov=app

# Async tests
pytest tests/ -v -s
```

## 📝 Логування

Структуроване JSON логування через structlog:

```python
from app.monitoring import log

log.info("event_name", user_id=123, status="ok")
log.error("error_event", error="msg", trace="...")
```

## 🔄 GitHub PR Верифікація

Процес:
1. Користувач сабмітить PR URL
2. Бот отримує webhook
3. Celery задача перевіряє PR через GitHub API
4. Якщо merged - автоматично approve
5. Виконується payout webhook

```python
# app/github_verifier.py
await verify_github_pr(pr_url)
# Returns: {verified: bool, merged: bool, author: str, message: str}
```

## 💰 Payout Integration

Для інтеграції зі Stripe/PayPal:

1. Налаштуй PAYOUT_WEBHOOK_URL в .env
2. Бот відправить POST з `{user_id, assignment_id, amount_cents}`
3. Твій payout сервіс обробить платіж

```bash
curl -X POST https://accounting.example/api/payouts \
  -H "X-Payout-Secret: $SECRET" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "assignment_id": 456, "amount_cents": 10000}'
```

## 🛠️ Troubleshooting

### Webhook не приймає

```bash
# 1. Перевір логи
docker-compose logs web | grep webhook

# 2. Перевір webhook URL
curl https://your-domain.com/health

# 3. Перевір WEBHOOK_SECRET
echo $WEBHOOK_SECRET
```

### DB migrations помилка

```bash
# 1. Перевір подключення
docker-compose exec db psql -U bot -d botdb

# 2. Скасуй останню міграцію
docker-compose exec web alembic downgrade -1

# 3. Заново
docker-compose exec web alembic upgrade head
```

### Celery воркер не крутиться

```bash
# 1. Логи
docker-compose logs worker

# 2. Перевір Redis
docker-compose exec redis redis-cli ping

# 3. Перезапусти
docker-compose restart worker
```

## 📚 Додатково

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Celery](https://docs.celeryproject.org/)
- [Telegram Bot API](https://core.telegram.org/bots)
- [Alembic](https://alembic.sqlalchemy.org/)

## 📄 Ліцензія

MIT

---

Made with ❤️ for PromptOps
